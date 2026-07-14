"""GeoNLA 两层简化模型，以及消融用扁平 MLP 基线。"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeoNLAModel(nn.Module):
    """
    简化版 Geo-Neural Layered Architecture。

    对应论文中的圈层同构思想：
    - physical_layer ≈ 物理地理圈层变换（W1）
    - human_layer ≈ 人地关系 / 人类圈涌现（W2）
    - time_modulation ≈ 时变耦合 W_k(t)
    - socio_modulation ≈ 外生社会经济驱动
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        output_dim: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.physical_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.time_modulation = nn.Parameter(torch.randn(hidden_dim) * 0.1)

        self.human_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.socio_modulation = nn.Parameter(torch.randn(output_dim) * 0.1)

    def forward(
        self,
        x_geo: torch.Tensor,
        time_factor: torch.Tensor,
        socio_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_geo : [B, input_dim]
        time_factor : [B, 1] 或 [B]，季节性标量，建议取值约 [-1, 1]
        socio_factor : [B, 1] 或 [B, output_dim]

        Returns
        -------
        y_human : [B, output_dim]
        h1 : [B, hidden_dim] 自然环境涌现层
        """
        if time_factor.ndim == 1:
            time_factor = time_factor.unsqueeze(-1)
        if socio_factor.ndim == 1:
            socio_factor = socio_factor.unsqueeze(-1)

        h1 = self.physical_layer(x_geo)
        # h1(t) = h1 * (1 + ΔW(t) * f(t))
        h1 = h1 * (1.0 + self.time_modulation * time_factor)

        y_human = self.human_layer(h1)
        y_human = y_human * (1.0 + self.socio_modulation * socio_factor)
        y_human = torch.clamp(y_human, 0.0, 1.0)

        return y_human, h1

    @property
    def W1(self) -> torch.Tensor:
        """物理地理耦合权重 [hidden_dim, input_dim]。"""
        return self.physical_layer[0].weight.detach()

    @property
    def W2(self) -> torch.Tensor:
        """人地关系耦合权重 [output_dim, hidden_dim]。"""
        return self.human_layer[0].weight.detach()


class FlatMLP(nn.Module):
    """
    消融基线：扁平 MLP，无地理层叠结构。

    - 将地理要素、时间因子、社会经济因子直接拼接为单一输入
    - 无圈层涌现、无乘性时变耦合
    - 隐藏维度与 GeoNLA 对齐，保证容量大致可比
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        output_dim: int = 6,
        dropout: float = 0.2,
        extra_dim: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.extra_dim = extra_dim

        flat_in = input_dim + extra_dim
        self.net = nn.Sequential(
            nn.Linear(flat_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_geo: torch.Tensor,
        time_factor: torch.Tensor,
        socio_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time_factor.ndim == 1:
            time_factor = time_factor.unsqueeze(-1)
        if socio_factor.ndim == 1:
            socio_factor = socio_factor.unsqueeze(-1)

        # 扁平化：时间/社会仅作附加特征，不做层间调制
        extras = []
        if self.extra_dim >= 1:
            extras.append(time_factor[:, :1])
        if self.extra_dim >= 2:
            extras.append(socio_factor[:, :1])
        x_flat = torch.cat([x_geo, *extras], dim=-1) if extras else x_geo

        y = self.net(x_flat)
        # 用第一隐藏层激活作为“伪涌现”便于可视化对比
        h = torch.relu(self.net[0](x_flat))
        return y, h


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"geonla", "geo_nla"}:
        return GeoNLAModel(**kwargs)
    if name in {"flat", "flatmlp", "flat_mlp"}:
        return FlatMLP(**kwargs)
    raise ValueError(f"未知模型: {name}")
