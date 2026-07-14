"""临夏盆地风格合成数据：内嵌地理因果链，便于验证层叠涌现。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config


@dataclass
class GeoNLADataset:
    """一次生成的完整数据集。"""

    X: np.ndarray          # [N, 12]
    y: np.ndarray          # [N, 6]
    time_factor: np.ndarray  # [N, 1]
    socio_factor: np.ndarray  # [N, 1]
    month: np.ndarray      # [N]
    coords: np.ndarray     # [N, 2] 伪空间坐标 (row, col)，便于后续扩展


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_linxia_synthetic(
    n_samples: int = config.N_SAMPLES,
    seed: int = config.SEED,
) -> GeoNLADataset:
    """
    按简化地理因果链生成合成样本：

    地形 → 气候 → 土壤 → 植被 → 人类活动
    并叠加季节项与社会经济外生项。
    """
    rng = np.random.default_rng(seed)

    # 伪空间网格坐标，使邻近样本具有相关性
    grid_side = int(np.ceil(np.sqrt(n_samples)))
    rows = np.repeat(np.arange(grid_side), grid_side)[:n_samples]
    cols = np.tile(np.arange(grid_side), grid_side)[:n_samples]
    coords = np.stack([rows, cols], axis=1).astype(np.float64)
    # 空间平滑噪声场
    spatial = (
        0.6 * np.sin(rows / grid_side * 2 * np.pi)
        + 0.4 * np.cos(cols / grid_side * 2 * np.pi)
    )

    # --- 地形（临夏高原谷地风格）---
    elev = _linxia_style_elev(rng, n_samples, spatial)
    slope = np.clip(0.02 * (elev - elev.mean()) + rng.normal(8, 4, n_samples), 0, 40)
    relief = np.clip(0.8 * slope + rng.normal(50, 20, n_samples), 10, 400)

    # --- 气候：受海拔与空间位置调制 ---
    temp = (
        config.LINXIA_TEMP_MEAN
        - 0.006 * (elev - config.LINXIA_ELEV_MEAN)
        + 1.5 * spatial
        + rng.normal(0, 1.2, n_samples)
    )
    precip = (
        config.LINXIA_PRECIP_MEAN
        - 0.15 * (elev - config.LINXIA_ELEV_MEAN)
        + 40 * spatial
        + rng.normal(0, 40, n_samples)
    )
    precip = np.clip(precip, 150, 900)
    aridity = np.clip((temp + 10) / (precip / 100.0 + 1e-3) + rng.normal(0, 0.3, n_samples), 0.2, 8)

    # --- 土壤：受气候与地形影响 ---
    soc = np.clip(
        25 - 0.4 * aridity + 0.02 * precip / 10 + rng.normal(0, 2, n_samples),
        2,
        60,
    )
    clay = np.clip(
        28 - 0.15 * slope + 0.5 * spatial * 10 + rng.normal(0, 4, n_samples),
        5,
        55,
    )
    field_cap = np.clip(
        0.15 * soc + 0.2 * clay + rng.normal(20, 3, n_samples),
        8,
        50,
    )

    # --- 植被：受气候、土壤影响 ---
    ndvi = _sigmoid(
        0.015 * precip - 0.2 * aridity + 0.03 * soc - 0.04 * slope + rng.normal(0, 0.4, n_samples)
    )
    ndvi_amp = np.clip(0.15 + 0.25 * ndvi + rng.normal(0, 0.05, n_samples), 0.05, 0.6)
    grass = np.clip(
        0.55 - 0.3 * ndvi + 0.01 * precip / 100 + rng.normal(0, 0.08, n_samples),
        0.05,
        0.9,
    )

    X = np.column_stack(
        [
            elev, slope, relief,
            temp, precip, aridity,
            soc, clay, field_cap,
            ndvi, ndvi_amp, grass,
        ]
    ).astype(np.float64)

    # 标准化输入特征（训练侧再确认一次）
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # --- 时间与社会经济因子 ---
    month = rng.integers(1, 13, size=n_samples)
    time_factor = np.sin(2 * np.pi * month / 12.0).astype(np.float64)[:, None]
    socio_factor = rng.uniform(0.2, 0.9, size=(n_samples, 1))

    # --- 标签按 GeoNLA 同构两阶段生成（使层叠+乘性调制成为正确归纳偏置）---
    # Stage-1: 地理要素 → 自然环境涌现 h1
    hidden = config.HIDDEN_DIM
    W1 = rng.normal(0, 0.35, size=(hidden, config.INPUT_DIM))
    # 注入可解释先验：若干涌现单元偏置到关键地理耦合
    W1[0, 0] = -1.2   # 高海拔
    W1[1, 4] = 1.1    # 降水
    W1[2, 5] = 1.0    # 干燥度
    W1[3, 9] = 1.0    # NDVI
    W1[4, 1] = -1.0   # 坡度
    b1 = rng.normal(0, 0.1, size=(hidden,))
    h1 = np.maximum(0.0, X_std @ W1.T + b1)

    # 时变耦合：乘性调制（扁平拼接难以等价表达）
    time_mod = rng.normal(0, 0.25, size=(hidden,))
    time_mod[:6] = np.array([0.8, 0.7, 0.6, 0.5, 0.2, 0.15])  # 前几维季节敏感
    h1_t = h1 * (1.0 + time_mod * time_factor)

    # Stage-2: 涌现层 → 人类活动
    W2 = rng.normal(0, 0.35, size=(config.OUTPUT_DIM, hidden))
    W2[0, 1] = 1.0    # 降水涌现 → 耕地
    W2[0, 3] = 0.8    # NDVI 涌现 → 耕地
    W2[1, 2] = 1.0    # 干燥涌现 → 灌溉
    W2[2, 4] = 0.9    # 平坦涌现 → 建设
    W2[3, 0] = -0.6
    W2[4, 4] = 0.7
    W2[5, 4] = 0.6
    b2 = rng.normal(0, 0.05, size=(config.OUTPUT_DIM,))
    y_base = _sigmoid(h1_t @ W2.T + b2)

    # 社会经济乘性调制
    socio_mod = np.array([0.15, 0.25, 0.55, 0.65, 0.70, 0.45])
    y = np.clip(
        y_base * (1.0 + socio_mod * socio_factor)
        + rng.normal(0, 0.02, size=y_base.shape),
        0.0,
        1.0,
    ).astype(np.float64)

    return GeoNLADataset(
        X=X_std,
        y=y,
        time_factor=time_factor,
        socio_factor=socio_factor,
        month=month.astype(np.int64),
        coords=coords,
    )


def _linxia_style_elev(rng: np.random.Generator, n: int, spatial: np.ndarray) -> np.ndarray:
    """围绕临夏均值生成海拔。"""
    elev = (
        config.LINXIA_ELEV_MEAN
        + config.LINXIA_ELEV_STD * (0.7 * spatial + 0.3 * rng.normal(0, 1, n))
        + rng.normal(0, 80, n)
    )
    return np.clip(elev, 1600, 3200)


def train_val_split(
    dataset: GeoNLADataset,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.SEED,
) -> tuple[GeoNLADataset, GeoNLADataset]:
    n = len(dataset.X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _subset(indices: np.ndarray) -> GeoNLADataset:
        return GeoNLADataset(
            X=dataset.X[indices],
            y=dataset.y[indices],
            time_factor=dataset.time_factor[indices],
            socio_factor=dataset.socio_factor[indices],
            month=dataset.month[indices],
            coords=dataset.coords[indices],
        )

    return _subset(train_idx), _subset(val_idx)


def make_loaders(
    train_set: GeoNLADataset,
    val_set: GeoNLADataset,
    batch_size: int = config.BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    def _to_loader(ds: GeoNLADataset, shuffle: bool) -> DataLoader:
        tensors = TensorDataset(
            torch.tensor(ds.X, dtype=torch.float32),
            torch.tensor(ds.time_factor, dtype=torch.float32),
            torch.tensor(ds.socio_factor, dtype=torch.float32),
            torch.tensor(ds.y, dtype=torch.float32),
        )
        return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)

    return _to_loader(train_set, True), _to_loader(val_set, False)
