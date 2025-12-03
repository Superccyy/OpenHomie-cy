# ... (版权信息省略) ...
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.him_estimator import HIMEstimator

# ==========================================
# 辅助函数：激活函数工厂
# ==========================================
def get_activation(act_name):
    """根据字符串名称返回 PyTorch 激活函数层"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
class RunningMeanStd:
    """
    动态计算输入数据的均值和方差，用于在线归一化 (Online Normalization)。
    这对 RL 的稳定性至关重要。
    """
    def __init__(self, shape, device):  # shape: 输入数据的维度
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        """增量式更新均值和方差"""
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        # Welford 算法的变体
        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    """归一化层封装"""
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # 评估/测试时 update=False，只使用已有的统计量进行归一化
        if update:  
            self.running_ms.update(x)
        # (x - mean) / std
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)
        return x

# ==========================================
# 核心类：HIMActorCritic
# ==========================================
class HIMActorCritic(nn.Module):
    is_recurrent = False # 声明这不是 RNN/LSTM
    
    def __init__(self,  num_actor_obs,            # Actor 输入总维度
                        num_critic_obs,           # Critic 输入总维度
                        num_one_step_obs,         # 单帧 Actor 观测维度
                        num_one_step_critic_obs,  # 单帧 Critic 观测维度
                        actor_history_length,     # Actor 历史帧数
                        critic_history_length,    # Critic 历史帧数
                        num_actions=19,           # 动作空间维度 (G1通常是12+手臂?)
                        actor_hidden_dims=[512, 256, 128],  # Actor MLP 隐藏层
                        critic_hidden_dims=[512, 256, 128], # Critic MLP 隐藏层
                        activation='elu',
                        init_noise_std=1.0,       # 初始探索噪声标准差
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments..." + str([key for key in kwargs.keys()]))
        super(HIMActorCritic, self).__init__()

        activation = get_activation(activation)
        # 保存各种维度配置
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_critic_obs = num_one_step_critic_obs
        self.actor_history_length = actor_history_length
        self.critic_history_length = critic_history_length
        
        # 计算本体感知数据的长度 = 历史长度 * 单帧长度
        self.actor_proprioceptive_obs_length = self.actor_history_length * self.num_one_step_obs
        self.critic_proprioceptive_obs_length = self.critic_history_length * self.num_one_step_critic_obs
        
        # 计算高度图数据点的数量 (如果有的话)
        # 总观测 - 本体感知 = 地形高度扫描点
        self.num_height_points = self.num_actor_obs - self.actor_proprioceptive_obs_length
        self.num_critic_height_points = self.num_critic_obs - self.critic_proprioceptive_obs_length
        self.actor_use_height = True if self.num_height_points > 0 else False
        self.num_actions = num_actions

        # 潜在向量维度 (Latent Dimension)
        self.dynamic_latent_dim = 32 # 动态特征 (由 Estimator 提取)
        self.terrain_latent_dim = 32 # 地形特征 (由 Terrain Encoder 提取)
        
        # --- 计算 Actor MLP 的输入维度 ---
        # 输入 = 当前单帧观测 + 指令(3维) + 动态特征(32) [+ 地形特征(32)]
        # 注意：这里的 "+3" 通常指 [vx, vy, yaw] 指令，但具体取决于 obs 结构
        if self.actor_use_height:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim + self.terrain_latent_dim
        else:
            mlp_input_dim_a = num_one_step_obs + 3 + self.dynamic_latent_dim
        
        mlp_input_dim_c = num_critic_obs

        # --- 1. Estimator (估计器) ---
        # 用于从历史观测中提取隐含的动态特征 (如摩擦力、负载、速度估计)
        # 这是一个典型的 RMA (Rapid Motor Adaptation) 架构
        self.estimator = HIMEstimator(temporal_steps=self.actor_history_length, 
                                      num_one_step_obs=self.num_one_step_obs, 
                                      num_height_points=0, 
                                      latent_dim=self.dynamic_latent_dim)
        
        # --- 2. Terrain Encoder (地形编码器) ---
        # 如果有高度扫描数据，用 MLP 压缩成 latent
        if self.actor_use_height:
            self.terrain_encoder = nn.Sequential(
                nn.Linear(self.num_one_step_obs + self.num_height_points, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.terrain_latent_dim),
            )

        # --- 3. Actor (策略网络) ---
        # 简单的 MLP
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # 最后一层通常没有激活函数，或者是 Tanh (如果是确定性策略)
                # 这里注释掉了 Tanh，说明输出范围由后续处理决定
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # --- 4. Critic (价值网络) ---
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1)) # 输出单一价值 V
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f'Estimator: {self.estimator.encoder}')
        if self.actor_use_height:
            print(f'Terrain Encoder: {self.terrain_encoder}')
        

        # 动作噪声参数 (可学习) - 用于 PPO 的随机探索
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        """权重初始化函数 (Orthogonal Initialization)"""
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ==========================================
    # 核心推理逻辑：构造 Actor 输入
    # ==========================================
    def update_distribution(self, obs_history):
        """
        根据观测历史计算动作分布 (均值和方差)。
        训练时使用。
        """
        with torch.no_grad():
            # 1. 运行 Estimator：从本体感知历史中提取 速度(vel) 和 动态特征(latent)
            # 这是 RMA 的核心：利用历史信息隐式估计环境参数
            vel, dynamic_latent = self.estimator(obs_history[:, 0:self.actor_proprioceptive_obs_length])
        
        if self.actor_use_height:
            # 2. 运行 Terrain Encoder：处理高度图
            terrain_latent = self.terrain_encoder(obs_history[:,-(self.num_height_points + self.num_one_step_obs):])
            # 3. 拼接所有特征
            # [当前帧观测, 估计速度, 动态Latent, 地形Latent]
            actor_input = torch.cat((obs_history[:,-(self.num_height_points + self.num_one_step_obs):-self.num_height_points], vel, dynamic_latent, terrain_latent), dim=-1)
        else:
            # [当前帧观测, 估计速度, 动态Latent]
            actor_input = torch.cat((obs_history[:,-self.num_one_step_obs:], vel, dynamic_latent), dim=-1)
        
        # 4. 运行 Actor MLP
        action_mean = self.actor(actor_input)
        
        # 5. 构建正态分布 (均值=网络输出, 方差=self.std)
        self.distribution = Normal(action_mean, action_mean*0. + self.std)

    def act(self, obs_history=None, **kwargs):
        """采样动作 (训练时包含随机性)"""
        self.update_distribution(obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ==========================================
    # 推理接口 (Deployment)
    # ==========================================
    def act_inference(self, obs_history, observations=None):
        """
        部署时使用的推理函数 (Deterministic, 无随机噪声)。
        导出 ONNX 时通常调用这个。
        """
        with torch.no_grad():
            # 1. Estimator
            vel, dynamic_latent = self.estimator(obs_history[:, 0:self.actor_proprioceptive_obs_length])
        
        if self.actor_use_height:
            terrain_latent = self.terrain_encoder(obs_history[:,-(self.num_height_points + self.num_one_step_obs):])
            actor_input = torch.cat((obs_history[:,-(self.num_height_points + self.num_one_step_obs):-self.num_height_points], vel, dynamic_latent, terrain_latent), dim=-1)
        else:
            # 拼接: [Current Obs, Estimated Velocity, Dynamic Latent]
            actor_input = torch.cat((obs_history[:,-self.num_one_step_obs:], vel, dynamic_latent), dim=-1)
        
        # 2. Actor
        action_mean = self.actor(actor_input)
        return action_mean # 直接返回均值作为动作

    def evaluate(self, critic_observations, **kwargs):
        """Critic 网络估值"""
        value = self.critic(critic_observations)
        return value
    
    def update_estimator(self, obs_history, next_critic_obs, lr=None):
        """
        更新 Estimator 网络。
        这通常通过监督学习的方式进行，目标是让 Estimator 的输出接近特权信息 (Privileged Info)。
        """
        return self.estimator.update(obs_history[:, 0:self.actor_proprioceptive_obs_length], next_critic_obs[:, 0:self.critic_proprioceptive_obs_length], lr=lr)
