# ... (版权信息省略) ...
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import HIMActorCritic
from rsl_rl.storage import HIMRolloutStorage

class HIMPPO:
    # 类型注解：actor_critic 必须是 HIMActorCritic 类的实例
    actor_critic: HIMActorCritic
    
    def __init__(self,
                 actor_critic,
                 use_flip = True,            # 是否启用镜像翻转（对称性利用的核心）
                 num_learning_epochs=1,      # 每次更新循环的 epoch 数
                 num_mini_batches=1,         # 每次更新的 mini-batch 数
                 clip_param=0.2,             # PPO 的裁剪参数 epsilon
                 gamma=0.998,                # 折扣因子
                 lam=0.95,                   # GAE (Generalized Advantage Estimation) 参数
                 value_loss_coef=1.0,        # 价值损失权重
                 entropy_coef=0.0,           # 熵正则化权重
                 learning_rate=1e-3,         # 学习率
                 max_grad_norm=1.0,          # 梯度裁剪阈值
                 use_clipped_value_loss=True,# 是否使用裁剪后的价值损失
                 schedule="fixed",           # 学习率调度策略
                 desired_kl=0.01,            # 目标 KL 散度 (用于自适应学习率)
                 device='cpu',               # 运行设备 (cuda/cpu)
                 symmetry_scale=1e-3,        # 对称性损失的权重系数 (重要参数!)
                 ):

        self.device = device
        self.use_flip = use_flip

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO 组件初始化
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # 数据存储器，稍后初始化
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # 两个 Transition 缓冲区：一个存原始数据，一个存镜像数据
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
        self.symmetry_scale = symmetry_scale
        
        # PPO 参数保存
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # 初始化经验回放池
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        """
        在环境中执行一步交互，同时处理原始观测和镜像观测。
        """
        # --- 1. 处理原始数据 ---
        # 计算动作分布、价值
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # 记录观测值
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        # --- 2. 处理镜像数据 (Symmetry) ---
        # [关键步骤]：对观测进行镜像翻转 (左变右，右变左)
        obs_sym = self.flip_g1_actor_obs(obs)
        critic_obs_sym = self.flip_g1_critic_obs(critic_obs)
        
        # 使用相同的网络对镜像观测进行推理
        self.transition_sym.actions = self.actor_critic.act(obs_sym).detach()
        self.transition_sym.values = self.actor_critic.evaluate(critic_obs_sym).detach()
        self.transition_sym.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition_sym.actions).detach()
        self.transition_sym.action_mean = self.actor_critic.action_mean.detach()
        self.transition_sym.action_sigma = self.actor_critic.action_std.detach()
        # 记录镜像观测
        self.transition_sym.observations = obs_sym
        self.transition_sym.critic_observations = critic_obs_sym
        
        # 返回给环境执行的动作 (必须是原始观测产生的动作)
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        """处理环境反馈的奖励和结束信号"""
        # 记录原始数据的奖励
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # 记录镜像数据的奖励 (奖励是对称的，假设环境也是对称的)
        next_critic_obs_sym = self.flip_g1_critic_obs(next_critic_obs)
        self.transition_sym.next_critic_observations = next_critic_obs_sym.clone()
        self.transition_sym.rewards = rewards.clone() # 镜像状态获得相同的奖励
        self.transition_sym.dones = dones
        
        # 处理超时 (Time out) 的情况，进行自举 (Bootstrapping)
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition_sym.rewards += self.gamma * torch.squeeze(self.transition_sym.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        
        # 将两份数据都存入 Buffer，实现数据增强 (Data Augmentation)
        self.storage.add_transitions(self.transition)
        self.storage.add_transitions(self.transition_sym)
        
        self.transition.clear()
        self.transition_sym.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        # 计算优势函数 (GAE)
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """PPO 更新主循环"""
        # ... (初始化统计变量) ...
        
        # 生成 mini-batch
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                
                # 1. 前向传播
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # 2. KL 散度计算与自适应学习率调整
                if self.desired_kl != None and self.schedule == 'adaptive':
                    # ... (标准 PPO 的 KL 调整逻辑) ...

                # 3. Estimator (估计器) 更新
                # 这是 Homie 论文中提到的 Teacher-Student 架构的一部分，或者是 RMA 架构
                if self.use_flip:
                    flipped_obs_batch = self.flip_g1_actor_obs(obs_batch)
                    flipped_next_critic_obs_batch = self.flip_g1_critic_obs(next_critic_obs_batch)
                    # 拼接原始数据和翻转数据一起训练估计器
                    estimator_update_obs_batch =  torch.cat((obs_batch, flipped_obs_batch), dim=0)
                    estimator_update_next_critic_obs_batch = torch.cat((next_critic_obs_batch, flipped_next_critic_obs_batch), dim=0)
                else:
                    estimator_update_obs_batch = obs_batch
                    estimator_update_next_critic_obs_batch = next_critic_obs_batch
                
                # 更新 Estimator 网络
                estimation_loss, swap_loss = self.actor_critic.update_estimator(estimator_update_obs_batch, estimator_update_next_critic_obs_batch, lr=self.learning_rate)
                
                # 4. 计算 PPO 损失 (Surrogate Loss)
                # ... (标准的 PPO Clip Loss 计算) ...

                # 5. 计算 Value Loss
                # ... (标准的 Value Loss 计算) ...
                
                # 6. [核心] 计算对称性损失 (Symmetry Loss)
                if self.use_flip:
                    flipped_critic_obs_batch = self.flip_g1_critic_obs(critic_obs_batch)
                    flipped_obs_batch = self.flip_g1_actor_obs(obs_batch) # 注意：这里重复计算了一次，可以优化
                    
                    # Actor 对称性损失：
                    # Loss = || Network(Flipped_Obs) - Flip(Network(Obs)) ||^2
                    # 意思是：输入翻转后的状态，网络输出的动作应该等于原始输出动作的翻转
                    actor_sym_loss = self.symmetry_scale * torch.mean(torch.sum(torch.square(
                        self.actor_critic.act_inference(flipped_obs_batch) - 
                        self.flip_g1_actions(self.actor_critic.act_inference(obs_batch))
                    ), dim=-1))
                    
                    # Critic 对称性损失：
                    # Loss = || Value(Flipped_Obs) - Value(Obs) ||^2
                    # 意思是：左右镜像的状态，其价值应该是相等的
                    critic_sym_loss = self.symmetry_scale * torch.mean(torch.square(
                        self.actor_critic.evaluate(flipped_critic_obs_batch) - 
                        self.actor_critic.evaluate(critic_obs_batch).detach()
                    ))
                    
                    # 总损失 = PPO损失 + Value损失 - 熵 + 对称损失
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + actor_sym_loss + critic_sym_loss
                else:
                    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # 7. 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ... (统计数据记录) ...

        # ... (清理与返回) ...

    # ==========================================
    # 辅助函数：状态和动作的镜像翻转
    # ==========================================
    
    def flip_g1_actor_obs(self, obs):
        """
        翻转 Actor 的观测向量 (针对 Unitree G1 机器人)
        逻辑：交换左右关节的位置/速度，翻转 Y 轴和 Yaw 轴的指令/角速度。
        """
        # 取出本体感知数据 (Proprioception)
        proprioceptive_obs = torch.clone(obs[:, :self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length])
        proprioceptive_obs = proprioceptive_obs.view(-1, self.actor_critic.actor_history_length, self.actor_critic.num_one_step_obs)
        
        flipped_proprioceptive_obs = torch.zeros_like(proprioceptive_obs)
        
        # --- 1. 翻转指令和基座状态 ---
        # x 不变, y 取反, yaw 取反
        flipped_proprioceptive_obs[:, :, 0] =  proprioceptive_obs[:, :, 0] # x command
        flipped_proprioceptive_obs[:, :, 1] = -proprioceptive_obs[:, :, 1] # y command
        flipped_proprioceptive_obs[:, :, 2] = -proprioceptive_obs[:, :, 2] # yaw command
        # ... (Roll 取反, Pitch 不变, Yaw 取反) ...
        
        # --- 2. 交换左右关节位置 (Joint Pos) ---
        # 假设 obs 索引 10~15 是左腿，16~21 是右腿 (或相反，需对照 G1 定义)
        # 这里代码似乎是：16~21(右) -> 10~15(左)，10~15(左) -> 16~21(右)
        # 同时注意某些关节需要取反（如 Roll 关节）
        flipped_proprioceptive_obs[:, :, 10] =  proprioceptive_obs[:, :, 16] # Right Hip Pitch -> Left Hip Pitch
        flipped_proprioceptive_obs[:, :, 11] = -proprioceptive_obs[:, :, 17] # Right Hip Roll -> -Left Hip Roll (镜像关系)
        # ... (其余关节类似处理) ...

        # --- 3. 交换左右关节速度 (Joint Vel) ---
        # 逻辑同上，只是索引偏移了 27 (关节数)
        # ...

        return flipped_proprioceptive_obs.view(-1, self.actor_critic.num_one_step_obs * self.actor_critic.actor_history_length).detach()
    
    def flip_g1_critic_obs(self, critic_obs):
        """翻转 Critic 的观测向量 (通常包含特权信息，逻辑同上)"""
        # ... (代码逻辑与 flip_g1_actor_obs 类似) ...
        return flipped_proprioceptive_obs.view(...).detach()
    
    def flip_g1_actions(self, actions):
        """
        翻转动作向量
        输入：[左腿动作, 右腿动作]
        输出：[右腿动作(经过符号调整), 左腿动作(经过符号调整)]
        """
        flipped_actions = torch.zeros_like(actions)
        # G1 动作顺序：0-5 右腿? 6-11 左腿? (需核对 URDF)
        # 这里将 index 6(左) 赋值给 index 0(右)，说明交换了左右
        # 并且某些自由度（如 Roll, Yaw）取了负号，符合镜像原理
        flipped_actions[:,  0] =  actions[:, 6]        # Left Hip Pitch -> Right Hip Pitch
        flipped_actions[:,  1] = -actions[:, 7]        # Left Hip Roll -> -Right Hip Roll
        flipped_actions[:,  2] = -actions[:, 8]        # Left Hip Yaw -> -Right Hip Yaw
        # ...
        return flipped_actions.detach()
