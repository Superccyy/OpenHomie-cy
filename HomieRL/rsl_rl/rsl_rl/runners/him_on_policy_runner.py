
import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import torch

import rsl_rl
from rsl_rl.algorithms import HIMPPO
from rsl_rl.modules import HIMActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

# ==========================================
# 核心类：HIMOnPolicyRunner (训练运行器)
# ==========================================
class HIMOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,       # 向量化环境实例 (包含并行运行的多个机器人)
                 train_cfg,         # 训练配置字典 (从 yaml 加载)
                 log_dir=None,      # 日志保存路径
                 device='cpu'):     # 运行设备 (cuda:0)

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # --- 1. 确定观测空间维度 ---
        # 区分 "普通观测" 和 "特权观测" (Privileged Obs, 用于 Critic)
        # 如果环境没有提供特权观测，则 Critic 和 Actor 使用相同的输入
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
            num_one_step_critic_obs = self.env.num_one_step_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
            num_one_step_critic_obs = self.env.num_one_step_obs
            
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        self.actor_history_length = self.env.actor_history_length
        self.critic_history_length = self.env.critic_history_length

        # --- 2. 初始化策略网络 (Actor-Critic) ---
        # 动态加载策略类 (通常是 HIMActorCritic)
        actor_critic_class = eval(self.cfg["policy_class_name"]) 
        actor_critic: HIMActorCritic = actor_critic_class( 
                                                        self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_one_step_obs,
                                                        num_one_step_critic_obs,
                                                        self.env.actor_history_length,
                                                        self.env.critic_history_length,
                                                        self.env.num_lower_dof, # 动作空间维度
                                                        **self.policy_cfg).to(self.device)

        # --- 3. 初始化算法 (PPO) ---
        # 动态加载算法类 (通常是 HIMPPO)
        alg_class = eval(self.cfg["algorithm_class_name"]) 
        self.alg: HIMPPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"] # 每个环境每轮采集多少步
        self.save_interval = self.cfg["save_interval"]         # 保存间隔

        # --- 4. 初始化数据存储 (Storage) ---
        # 为 PPO 算法分配显存缓冲区
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_lower_dof])

        # 日志相关
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # 重置环境，准备开始
        _, _ = self.env.reset()
    
    # ==========================================
    # 主训练循环 (Learn Loop)
    # ==========================================
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # --- 1. 初始化 Logger (WandB / Tensorboard) ---
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "wandb")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
            
        # 随机化初始步数，打破环境同步 (防止所有机器人同时 Reset)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        # 获取初始观测
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        # 切换到训练模式
        self.alg.actor_critic.train() 

        # 统计数据缓冲区
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        # --- 循环开始：迭代更新 ---
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # === Phase 1: 数据收集 (Rollout) ===
            # 这一步是在与仿真环境交互，不更新网络权重，所以用 inference_mode 加速
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # 1. 策略网络决策
                    actions = self.alg.act(obs, critic_obs)
                    
                    # 2. 仿真环境执行一步
                    obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    # 3. 处理终止状态 (Timeout / Termination)
                    # PPO 需要正确的“下一个状态价值”来计算优势，对于终止状态需特殊处理
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)
                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    # 4. 存入 PPO 缓冲区 (同时处理对称性镜像)
                    self.alg.process_env_step(rewards, dones, infos, next_critic_obs)
                
                    # 5. 记录 Log 信息 (Reward, Episode Length)
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # 仅当环境重置时，才记录这一轮的总奖励和长度
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # === Phase 2: 网络更新 (Learning) ===
                start = stop
                # 1. 计算 GAE (优势函数)
                self.alg.compute_returns(critic_obs)
                
            # 2. 执行 PPO 更新 (包含对称性 Loss)
            mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss, mean_actor_sym_loss, mean_critic_sym_loss = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            
            # === Phase 3: 日志与保存 ===
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            self.current_learning_iteration = it
        
        # 训练结束，保存最终模型
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    # ==========================================
    # 辅助函数：日志记录
    # ==========================================
    def log(self, locs, width=80, pad=35):
        # 计算 FPS、平均奖励等指标并写入 Tensorboard
        # ... (代码略，主要是格式化打印) ...
        pass

    # ==========================================
    # 辅助函数：保存与加载
    # ==========================================
    def save(self, path, infos=None):
        # 保存网络权重、优化器状态、迭代次数
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration + 1,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        # 加载检查点
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        # 获取用于部署的推理策略 (eval模式)
        self.alg.actor_critic.eval() 
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
