"""GeoNLA vs 扁平 MLP 消融对比。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import config
from data import generate_linxia_synthetic, make_loaders, train_val_split
from model import build_model, count_parameters
from train import predict_metrics, set_seed, train_one_model
from visualize import plot_ablation_comparison


def run_ablation(source: str = "synthetic") -> dict:
    print("=" * 50)
    print(f"消融实验：GeoNLA vs FlatMLP | data={source}")
    print("=" * 50)

    set_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if source == "synthetic":
        full = generate_linxia_synthetic(n_samples=config.N_SAMPLES, seed=config.SEED)
    else:
        from data_raster import load_linxia_raster_dataset

        full = load_linxia_raster_dataset()

    train_set, val_set = train_val_split(full)
    train_loader, val_loader = make_loaders(train_set, val_set)

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

    model_kwargs = dict(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    )

    results = {}
    histories = {}

    for name in ("geonla", "flatmlp"):
        print(f"\n>>> 训练 {name}")
        set_seed(config.SEED)  # 同一初始化种子，公平对比
        model = build_model(name, **model_kwargs)
        model, history, best_val = train_one_model(
            model, train_loader, val_loader, log_name=name
        )
        metrics = predict_metrics(model, val_loader)
        ckpt_path = config.OUTPUT_DIR / f"{name}_checkpoint.pth"
        torch.save(
            {
                "model_name": name,
                "model_state_dict": model.state_dict(),
                "history": history,
                "best_val_loss": best_val,
                "val_metrics": metrics,
                "model_config": model_kwargs,
                "n_params": count_parameters(model),
            },
            ckpt_path,
        )
        # 兼容主可视化脚本
        if name == "geonla":
            torch.save(
                {
                    "model_name": name,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                    "best_val_loss": best_val,
                    "val_metrics": metrics,
                    "model_config": model_kwargs,
                    "train_config": {"source": source, "seed": config.SEED},
                },
                config.CHECKPOINT_PATH,
            )

        results[name] = {
            "best_val_loss": best_val,
            "n_params": count_parameters(model),
            "metrics": metrics,
            "checkpoint": str(ckpt_path),
        }
        histories[name] = history
        print(
            f"  {name}: val_mse={best_val:.4f} "
            f"rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
            f"r2={metrics['r2_mean']:.4f} params={count_parameters(model)}"
        )

    # 相对提升（以 FlatMLP 为基线）
    base = results["flatmlp"]["metrics"]
    geo = results["geonla"]["metrics"]
    lift = {
        "rmse_improve_pct": (base["rmse"] - geo["rmse"]) / (base["rmse"] + 1e-12) * 100,
        "mae_improve_pct": (base["mae"] - geo["mae"]) / (base["mae"] + 1e-12) * 100,
        "r2_improve_abs": geo["r2_mean"] - base["r2_mean"],
        "mse_improve_pct": (
            (results["flatmlp"]["best_val_loss"] - results["geonla"]["best_val_loss"])
            / (results["flatmlp"]["best_val_loss"] + 1e-12)
            * 100
        ),
    }
    summary = {
        "source": source,
        "models": results,
        "geonla_vs_flatmlp": lift,
        "data": str(data_path),
    }

    out_json = config.OUTPUT_DIR / f"ablation_results_{source}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # 兼容旧文件名
    if source == "synthetic":
        with open(config.OUTPUT_DIR / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    fig_dir = config.FIG_DIR if source == "synthetic" else config.FIG_DIR / "raster"
    fig_paths = plot_ablation_comparison(histories, results, fig_dir)
    print("\n消融完成:")
    print(f"  GeoNLA 相对 FlatMLP: RMSE↓ {lift['rmse_improve_pct']:.2f}%")
    print(f"  结果: {out_json}")
    for p in fig_paths:
        print(f"  图: {p}")
    return summary


if __name__ == "__main__":
    run_ablation("synthetic")
