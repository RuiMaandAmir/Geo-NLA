"""GeoNLA 训练入口：合成数据 / 栅格数据 → 训练 → 保存 checkpoint。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import config
from data import (
    GeoNLADataset,
    generate_linxia_synthetic,
    make_loaders,
    train_val_split,
)
from model import GeoNLAModel, build_model, count_parameters


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion) -> float:
    model.eval()
    total, n = 0.0, 0
    for x_geo, t_factor, s_factor, y_true in loader:
        y_pred, _ = model(x_geo, t_factor, s_factor)
        loss = criterion(y_pred, y_true)
        total += loss.item() * x_geo.size(0)
        n += x_geo.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict_metrics(model: nn.Module, loader) -> dict[str, float]:
    model.eval()
    preds, trues = [], []
    for x_geo, t_factor, s_factor, y_true in loader:
        y_pred, _ = model(x_geo, t_factor, s_factor)
        preds.append(y_pred.cpu().numpy())
        trues.append(y_true.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    # 逐维 R2；近常数标签维记为 nan，不参与均值
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2_per = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = ss_tot > 1e-8
    r2_per[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    r2_mean = float(np.nanmean(r2_per)) if np.any(valid) else float("nan")
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2_mean": r2_mean,
        "r2_per_dim": [None if np.isnan(v) else float(v) for v in r2_per],
    }


def train_one_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = config.EPOCHS,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
    log_name: str = "model",
) -> tuple[nn.Module, dict[str, list[float]], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    print(f"  [{log_name}] params={count_parameters(model)}")
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x_geo, t_factor, s_factor, y_true in train_loader:
            y_pred, _ = model(x_geo, t_factor, s_factor)
            loss = criterion(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * x_geo.size(0)
            n += x_geo.size(0)

        train_loss = running / max(n, 1)
        val_loss = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  [{log_name}] Epoch [{epoch:3d}/{epochs}] "
                f"train={train_loss:.4f} val={val_loss:.4f} lr={lr_now:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def load_dataset(source: str = "synthetic") -> GeoNLADataset:
    """source: synthetic | raster"""
    if source == "synthetic":
        return generate_linxia_synthetic(n_samples=config.N_SAMPLES, seed=config.SEED)
    if source == "raster":
        from data_raster import load_linxia_raster_dataset

        return load_linxia_raster_dataset()
    raise ValueError(f"未知数据源: {source}")


def train(
    model_name: str = "geonla",
    source: str = "synthetic",
    checkpoint_path: Path | None = None,
) -> tuple[Any, dict]:
    print("=" * 50)
    print(f"GeoNLA Demo 训练 | model={model_name} | data={source}")
    print("=" * 50)

    set_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH

    print("\n[1/4] 加载数据...")
    full = load_dataset(source)
    train_set, val_set = train_val_split(full)
    train_loader, val_loader = make_loaders(train_set, val_set)
    print(f"  样本总数: {len(full.X)}")
    print(f"  训练 / 验证: {len(train_set.X)} / {len(val_set.X)}")
    print(f"  X: {full.X.shape}, y: {full.y.shape}")

    print("\n[2/4] 初始化模型...")
    model_kwargs = dict(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    )
    model = build_model(model_name, **model_kwargs)

    print("\n[3/4] 开始训练...")
    model, history, best_val = train_one_model(
        model, train_loader, val_loader, log_name=model_name
    )
    metrics = predict_metrics(model, val_loader)

    print("\n[4/4] 保存结果...")
    checkpoint = {
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_loss": best_val,
        "val_metrics": metrics,
        "model_config": model_kwargs,
        "train_config": {
            "epochs": config.EPOCHS,
            "lr": config.LR,
            "batch_size": config.BATCH_SIZE,
            "n_samples": len(full.X),
            "seed": config.SEED,
            "source": source,
        },
    }
    torch.save(checkpoint, checkpoint_path)

    data_path = config.OUTPUT_DIR / f"{source}_data.npz"
    np.savez(
        data_path,
        X=full.X,
        y=full.y,
        time_factor=full.time_factor,
        socio_factor=full.socio_factor,
        month=full.month,
        coords=full.coords,
    )

    meta = {
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "best_val_loss": best_val,
        "val_metrics": metrics,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }
    with open(config.OUTPUT_DIR / f"train_meta_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 兼容旧路径
    if model_name == "geonla":
        with open(config.OUTPUT_DIR / "train_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  最优验证损失: {best_val:.4f}")
    print(f"  val RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2_mean']:.4f}")
    print(f"  checkpoint: {checkpoint_path}")
    return model, history


if __name__ == "__main__":
    train()
