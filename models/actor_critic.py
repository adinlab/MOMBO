import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple

from models.nets import MMDetLayer


# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist


class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values

class MMCriticBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [MMDetLayer(in_dim, out_dim)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [MMDetLayer(hidden_dims[-1], output_dim, isoutput=True)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, means: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        return self.model((means, vars))
    
class MMCritic(Critic):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__(backbone, device)
        self.last = MMDetLayer(backbone.output_dim, 1, isoutput=True).to(device)

    def forward(
        self,
        obs_mean: Union[np.ndarray, torch.Tensor],
        action_mean: Union[np.ndarray, torch.Tensor],
        obs_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
        action_var: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        means = torch.cat([obs_mean, action_mean], dim=-1).to(self.device).to(torch.float32)
        if obs_var is None:
            obs_var = torch.zeros_like(obs_mean)
        if action_var is None:
            action_var = torch.zeros_like(action_mean)
            
        vars = torch.cat([obs_var, action_var], dim=-1).to(self.device).to(torch.float32)
        
        logits = self.backbone(means, vars)
        q_mean, q_var = self.last(logits)
        return q_mean, q_var