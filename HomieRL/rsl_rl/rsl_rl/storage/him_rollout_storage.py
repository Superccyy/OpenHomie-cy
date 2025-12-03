# ... (版权信息省略) ...

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class HIMRolloutStorage:
    # ==========================================
    # 内部类：Transition (单步数据容器)
    # ==========================================
    class Transition:
        """
        用于临时存储单一步骤中收集到的所有数据。
        可以理解为一个结构体 (Struct)。
        """
        def __init__(self):
            self.observations = None        # 当前观测 (Actor)
            self.critic_observations = None # 当前观测 (Critic/特权)
            self.actions = None             # 执行的动作
            self.rewards = None             # 获得的奖励
            self.dones = None               # 是否结束标志
            self.values = None              # 价值估计 (V)
            self.actions_log_prob = None    # 动作对数概率 (用于PPO)
            self.action_mean = None         # 动作分布均值
            self.action_sigma = None        # 动作分布标准差
            self.next_critic_observations = None # 下一步的 Critic 观测
        
        def clear(self):
            self.__init__() # 清空数据

    # ==========================================
    # 初始化函数
    # ==========================================
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device='cpu'):
        """
        初始化 Rollout Storage。
        
        Args:
            num_envs: 并行环境数量 (例如 4096)
            num_transitions_per_env: 每个环境采集多少步 (例如 24)
            obs_shape: Actor 观测维度
            privileged_obs_shape: Critic 观测维度
            actions_shape: 动作维度 (例如 12)
        """
        self.device = device
        
        # [关键]：容量翻倍！
        # 因为 Homie 使用了对称性数据增强 (Symmetry Augmentation)，
        # 每一帧数据都会生成一个镜像数据，所以 Buffer 大小需要是原来的 2 倍。
        num_transitions_per_env *= 2 
        
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # --- 核心数据张量 (预分配显存) ---
        # 维度通常是 [总步数, 环境数, 特征维度]
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        
        # 如果 Critic 观测不同于 Actor，则单独存储
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
            self.next_privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
            self.next_privileged_observations = None
            
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # --- PPO 专用数据 ---
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) # V(s)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) # 实际回报 (GAE target)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) # 优势函数 A(s,a)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0 # 当前写入指针

    # ==========================================
    # 数据写入
    # ==========================================
    def add_transitions(self, transition: Transition):
        """将单步数据写入 Buffer"""
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
            
        # 使用 copy_() 进行原地显存拷贝，避免频繁内存分配
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: 
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        if self.next_privileged_observations is not None: 
            self.next_privileged_observations[self.step].copy_(transition.next_critic_observations)
            
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        
        self.step += 1

    def clear(self):
        """重置缓冲区指针"""
        self.step = 0

    # ==========================================
    # GAE 计算 (核心算法)
    # ==========================================
    def compute_returns(self, last_values, gamma, lam):
        """
        计算广义优势估计 (Generalized Advantage Estimation, GAE)。
        用于减少策略梯度估计的方差。
        """
        # 由于使用了对称性增强，buffer 里存了双倍数据 (原始 + 镜像)
        # 这里似乎将它们视为一个更长的序列进行处理
        num_transitions_per_env = self.num_transitions_per_env // 2
        advantage = 0
        
        # 这里的 view 操作可能是在将原始数据和镜像数据分开处理，或者合并处理
        # 代码逻辑是：将 buffer 视为 [steps, 2(原始+镜像), envs, dims]
        resize = lambda x: x.view(num_transitions_per_env, 2, -1, 1)
        
        # 逆序遍历 (从最后一步往前算)
        for step in reversed(range(num_transitions_per_env)):
            if step == num_transitions_per_env - 1:
                next_values = last_values # 最后一步的价值由 Critic 估计
            else:
                next_values = resize(self.values)[step + 1]
            
            # TD Error (Temporal Difference Error)
            # delta = r + gamma * V(s') * (1-done) - V(s)
            next_is_not_terminal = 1.0 - resize(self.dones)[step].float()
            delta = resize(self.rewards)[step] + next_is_not_terminal * gamma * next_values - resize(self.values)[step]
            
            # GAE 公式: A = delta + gamma * lambda * A_next
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            
            # Return = Advantage + Value
            resize(self.returns)[step] = advantage + resize(self.values)[step]
            
        # 计算优势函数并归一化 (Standardization)
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        """计算平均回合奖励和长度 (用于日志记录)"""
        done = self.dones
        done[-1] = 1 # 强制最后一步结束，方便计算
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    # ==========================================
    # Mini-Batch 生成器 (用于 PPO Update)
    # ==========================================
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        生成用于 PPO 更新的小批次数据。
        会将所有环境、所有步数的数据打乱 (Shuffle)。
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        
        # 生成随机索引，打乱数据
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        # 展平数据: [steps, envs, dim] -> [steps*envs, dim]
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
            next_critic_observations = self.next_privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
            next_critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # 多次遍历 (Epochs)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                # 提取 Mini-Batch
                obs_batch = observations[batch_idx]
                next_critic_observations_batch = next_critic_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                
                # Yield 返回生成器
                yield obs_batch, critic_observations_batch, actions_batch, next_critic_observations_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
