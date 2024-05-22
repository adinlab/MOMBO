import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from copy import deepcopy
from collections import defaultdict
from policies import BasePolicy
from dynamics import BaseDynamics


class MOMBOPolicy(BasePolicy):
    """
    Deterministic Uncertainty Propagation for Improved Model-Based Offline Reinforcement Learning
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        actor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        deterministic_backup: bool = False,
        max_q_backup: bool = False
    ) -> None:

        super().__init__()
        self.dynamics = dynamics
        self.actor = actor
        self.critics = critics
        self.critics_old = deepcopy(critics)
        self.critics_old.eval()

        self.actor_optim = actor_optim
        self.critics_optim = critics_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._penalty_coef = penalty_coef
        self._deteterministic_backup = deterministic_backup
        self._max_q_backup = max_q_backup

    def train(self) -> None:
        self.actor.train()
        self.critics.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critics.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
    
    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()
    
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:

        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss
        obs_variances = np.zeros_like(observations)
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, next_observation_variances, rewards, terminals, info = self.dynamics.step(observations, actions)

            rollout_transitions["obss"].append(observations)
            rollout_transitions['obs_variances'].append(obs_variances)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions['next_obs_variances'].append(next_observation_variances)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards)
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations[nonterm_mask]
            obs_variances = next_observation_variances[nonterm_mask]
        
        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
    
    @ torch.no_grad()
    def get_q_target_for_real(self, rewards: torch.Tensor, terminals: torch.Tensor, next_obss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        next_actions, next_log_probs = self.actforward(next_obss)
        next_qss = [critic_old(next_obss, next_actions, torch.zeros_like(next_obss)) for critic_old in self.critics_old]
        next_q_means = torch.stack([qs[0] for qs in next_qss], 0)
        next_q = torch.min(next_q_means, 0)[0]
        if not self._deteterministic_backup:
            next_q -= self._alpha * next_log_probs
            
        target_q = rewards + self._gamma * (1 - terminals) * next_q
        return target_q, torch.zeros_like(target_q)

    @ torch.no_grad()
    def get_q_target_for_fake(self, obss: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = obss.shape[0]
        num_elites = self.dynamics.model.num_elites
        
        next_obs_means, next_obs_variances, rewards, terminals, _ = self.dynamics.torch_step(obss, actions)
        
        next_obs_means = next_obs_means.view(num_elites * batch_size, -1)
        next_obs_variances = next_obs_variances.view(num_elites * batch_size, -1)
        
        next_actions, next_log_probs = self.actforward(next_obs_means)
            
        next_qss = [critic_old(next_obs_means, next_actions, next_obs_variances) for critic_old in self.critics_old]
        next_q_means = torch.stack([qs[0] for qs in next_qss], 0)
        next_q_variances = torch.stack([qs[1] for qs in next_qss], 0)
        
        idxs = torch.min(next_q_means, 0)[1].squeeze()
        next_q_mean = next_q_means[idxs, torch.arange(num_elites * batch_size)]
        next_q_variance = next_q_variances[idxs, torch.arange(num_elites * batch_size)]
        
        if not self._deteterministic_backup:
            next_q_mean -= self._alpha * next_log_probs
            
        next_q_mean = next_q_mean.view(num_elites, batch_size, 1)
        next_q_variance = next_q_variance.view(num_elites, batch_size, 1)
            
        target_q = rewards + self._gamma * (1 - terminals * 1.0) * (next_q_mean)
        
        mean_target_q = target_q.mean(0)
        penalty = target_q.std(0)
        target_q = mean_target_q - self._penalty_coef * penalty
        return target_q, penalty
    
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        
        obss, actions = mix_batch["observations"], mix_batch["actions"]
        obs_variances = mix_batch["observation_variances"]
        batch_size = obss.shape[0]

        # update critic
        qss = [critic(
                obss, actions
                ) for critic in self.critics]
        q_means = torch.stack([qs[0] for qs in qss], 0)
        
        with torch.no_grad():
            real_target_q, real_penalty = self.get_q_target_for_real(real_batch["rewards"], real_batch["terminals"], real_batch["next_observations"])
            fake_target_q, fake_penalty = self.get_q_target_for_fake(fake_batch["observations"], fake_batch["actions"])
            target_q = torch.cat([real_target_q, fake_target_q], 0)
            penalty = torch.cat([real_penalty, fake_penalty], 0)
            target_q = torch.clamp(target_q, 0, None)

        critic_loss = ((q_means - target_q) ** 2).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        a, log_probs = self.actforward(obss)
        qas = torch.cat([critic(obss, a, obs_variances)[0] for critic in self.critics], 1) # use mean
        actor_loss = -torch.min(qas, 1)[0].mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "penalty": penalty.mean().item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result