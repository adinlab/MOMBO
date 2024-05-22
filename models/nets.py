import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Dict, List, Union, Tuple, Optional


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss


class MMDetLayer(nn.Module):
    def __init__(self, in_features, out_features, isoutput=False):
        super(MMDetLayer, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.isoutput = isoutput
        self.EPS = 1e-16

        self.mean_layer = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        self.mean_layer.reset_parameters()

    def cdf(self, x, mu=0.0, sig=1.0):
        return 0.5 * (1 + torch.erf((x - mu) / (sig * math.sqrt(2))))

    def pdf(self, x, mu=0.0, sig=1.0):
        return (1 / (math.sqrt(2 * math.pi) * sig)) * torch.exp(
            -0.5 * ((x - mu) / sig).pow(2)
        )

    def relu_moments(self, mu, var):
        zeros = var.eq(0)
        sig = var.clamp(self.EPS).sqrt()
        alpha = mu / sig.clamp(self.EPS)
        cdf = self.cdf(alpha)
        pdf = self.pdf(alpha)
        relu_mean = mu * cdf + sig * pdf
        relu_var = (sig.pow(2) + mu.pow(2)) * cdf + mu * sig * pdf - relu_mean.pow(2)

        relu_mean = mu * (mu > 0) * zeros + relu_mean * ~zeros
        relu_var = relu_var.clamp(self.EPS)
        relu_var = torch.zeros_like(relu_var) * zeros + relu_var * ~zeros
        return relu_mean, relu_var

    def forward(self, input):
        mu_h, var_h = input
        if var_h is None:
            var_h = torch.zeros_like(mu_h)

        mu_f = self.mean_layer(mu_h)
        var_f = F.linear(var_h, self.mean_layer.weight.pow(2))

        # compute relu moments if it is not an output layer
        if not self.isoutput:
            return self.relu_moments(mu_f, var_f)
        else:
            return mu_f, var_f

    def __repr__(self):
        return (
            self.__class__.__name__
            + f" ({str(self.n_in)} -> {self.n_out}, isoutput={self.isoutput})"
        )
