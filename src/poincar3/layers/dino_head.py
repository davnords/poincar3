import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm


class DINOHead(nn.Module):
    """DINO/iBOT-style prototype head: MLP bottleneck -> L2-normalize -> weight-normalized linear.

    Matches the reference implementation in `DINOv2's `dinov2/layers/dino_head.py`.
    The weight-normalized last layer's magnitude is initialized to 1 (only its
    direction is learned from there) -- the standard DINO/iBOT trick for
    stabilizing the scale of the prototype logits.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 2048,
        bottleneck_features: int = 256,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        num_layers = max(num_layers, 1)
        if num_layers == 1:
            self.mlp = nn.Linear(in_features, bottleneck_features)
        else:
            layers = [nn.Linear(in_features, hidden_features), nn.GELU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_features, hidden_features), nn.GELU()]
            layers.append(nn.Linear(hidden_features, bottleneck_features))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = weight_norm(nn.Linear(bottleneck_features, out_features, bias=False))
        self.last_layer.parametrizations.weight.original0.data.fill_(1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2, eps=1e-6)
        return self.last_layer(x)
