"""GeoNLA 可视化：生成 5 张关键图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

import config
from model import GeoNLAModel

# Windows 优先中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_bundle(
    checkpoint_path: Path | None = None,
    data_path: Path | None = None,
):
    checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH
    data_path = data_path or (config.OUTPUT_DIR / "synthetic_data.npz")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = GeoNLAModel(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = np.load(data_path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    time_factor = torch.tensor(data["time_factor"], dtype=torch.float32)
    socio_factor = torch.tensor(data["socio_factor"], dtype=torch.float32)
    return model, ckpt, X, y, time_factor, socio_factor


def plot_model_architecture(fig_dir: Path) -> Path:
    """图1：模型架构示意。"""
    fig, ax = plt.subplots(figsize=(11, 5))
    layers = [
        ("地理要素输入层\n(12维)", 0.18, "#3B82F6"),
        ("自然环境涌现层\n(32维)", 0.50, "#10B981"),
        ("人类活动涌现层\n(6维)", 0.82, "#EF4444"),
    ]
    for label, x, color in layers:
        ax.add_patch(plt.Circle((x, 0.55), 0.09, color=color, alpha=0.75, zorder=2))
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=10, zorder=3)

    ax.annotate(
        "",
        xy=(0.40, 0.55),
        xytext=(0.28, 0.55),
        arrowprops=dict(arrowstyle="->", lw=2, color="#64748B"),
    )
    ax.annotate(
        "",
        xy=(0.72, 0.55),
        xytext=(0.60, 0.55),
        arrowprops=dict(arrowstyle="->", lw=2, color="#64748B"),
    )
    ax.text(0.34, 0.72, "W1 物理地理耦合", ha="center", fontsize=10, style="italic")
    ax.text(0.66, 0.72, "W2 人地关系耦合", ha="center", fontsize=10, style="italic")
    ax.text(
        0.50,
        0.28,
        "时间调制 f(t)  ·  社会经济调制 g(s)",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("GeoNLA 两层架构示意图", fontsize=14, pad=12)
    out = fig_dir / "fig1_model_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def plot_emergence_tsne(model, X, time_factor, socio_factor, fig_dir: Path) -> Path:
    """图2：涌现层 t-SNE。"""
    _, h1 = model(X, time_factor, socio_factor)
    h1_np = h1.numpy()
    # 用海拔代理着色，观察地理结构是否保留
    elev_proxy = X[:, 0].numpy()

    perplexity = min(30, max(5, len(h1_np) // 4))
    emb = TSNE(
        n_components=2,
        random_state=config.SEED,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(h1_np)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=elev_proxy, cmap="terrain", s=40, alpha=0.75)
    plt.colorbar(sc, ax=ax, label="标准化海拔（输入代理）")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("自然环境涌现层 t-SNE\n（颜色=海拔，观察地理相似性聚类）")
    out = fig_dir / "fig2_emergence_tsne.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def plot_prediction_comparison(model, X, time_factor, socio_factor, y_true, fig_dir: Path) -> Path:
    """图3：预测 vs 真实。"""
    y_pred, _ = model(X, time_factor, socio_factor)
    y_pred_np = y_pred.numpy()
    y_true_np = y_true.numpy()

    rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
    mae = float(np.mean(np.abs(y_pred_np - y_true_np)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, (ax, name) in enumerate(zip(axes.flatten(), config.OUTPUT_NAMES)):
        ax.scatter(y_true_np[:, i], y_pred_np[:, i], s=18, alpha=0.55, c="#2563EB")
        ax.plot([0, 1], [0, 1], "r--", lw=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.set_title(name)
        ax.grid(alpha=0.25)

    fig.suptitle(f"预测值 vs 真实值  |  RMSE={rmse:.4f}, MAE={mae:.4f}", fontsize=13)
    out = fig_dir / "fig3_prediction_comparison.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_weight_heatmaps(model: GeoNLAModel, fig_dir: Path) -> Path:
    """图4：W1 / W2 权重热力图。"""
    W1 = model.W1.cpu().numpy()
    W2 = model.W2.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        W1,
        ax=axes[0],
        cmap="RdBu_r",
        center=0,
        xticklabels=config.INPUT_NAMES,
        yticklabels=False,
        cbar_kws={"label": "权重"},
    )
    axes[0].set_title("W1：物理地理耦合 (隐藏×输入)")
    axes[0].set_xlabel("地理要素")
    axes[0].set_ylabel("涌现特征")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)

    sns.heatmap(
        W2,
        ax=axes[1],
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=config.OUTPUT_NAMES,
        cbar_kws={"label": "权重"},
    )
    axes[1].set_title("W2：人地关系耦合 (输出×隐藏)")
    axes[1].set_xlabel("涌现特征")
    axes[1].set_ylabel("人类活动")

    out = fig_dir / "fig4_weight_heatmaps.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def plot_temporal_modulation(model, X, fig_dir: Path) -> Path:
    """图5：固定地理条件，扫描 12 个月时间调制。"""
    x_sample = X[0:1]
    months = np.arange(1, 13)
    time_vals = np.sin(2 * np.pi * months / 12.0).astype(np.float32)
    time_t = torch.tensor(time_vals[:, None])
    socio_t = torch.full((12, 1), 0.5)
    x_rep = x_sample.repeat(12, 1)
    y_pred, _ = model(x_rep, time_t, socio_t)
    y_np = y_pred.numpy()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, (ax, name) in enumerate(zip(axes.flatten(), config.OUTPUT_NAMES)):
        ax.plot(months, y_np[:, i], "o-", lw=2, ms=6, color="#0F766E")
        ax.set_xticks(months)
        ax.set_xticklabels([f"{m}月" for m in months], fontsize=8)
        ax.set_xlabel("月份")
        ax.set_ylabel("预测值")
        ax.set_title(name)
        ax.grid(alpha=0.25)

    fig.suptitle("时间调制效果：固定地理与社会经济条件", fontsize=13)
    out = fig_dir / "fig5_temporal_modulation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_loss_curves(ckpt: dict, fig_dir: Path) -> Path:
    """附加：训练/验证损失曲线。"""
    hist = ckpt["history"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["train_loss"], label="train", color="#2563EB")
    ax.plot(hist["val_loss"], label="val", color="#DC2626")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("训练过程损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    out = fig_dir / "fig0_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_ablation_comparison(
    histories: dict,
    results: dict,
    fig_dir: Path | None = None,
) -> list[Path]:
    """消融对比图：损失曲线 + 指标柱状图 + 分维 R2。"""
    fig_dir = fig_dir or config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # 图A：验证损失曲线对比
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"geonla": "#0F766E", "flatmlp": "#B45309"}
    for name, hist in histories.items():
        ax.plot(
            hist["val_loss"],
            label=f"{name} val",
            color=colors.get(name, None),
            lw=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE")
    ax.set_title("消融对比：验证损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    out_a = fig_dir / "fig6_ablation_val_loss.png"
    fig.tight_layout()
    fig.savefig(out_a, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_a)

    # 图B：RMSE / MAE / R2
    names = list(results.keys())
    rmse = [results[n]["metrics"]["rmse"] for n in names]
    mae = [results[n]["metrics"]["mae"] for n in names]
    r2 = [results[n]["metrics"]["r2_mean"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, values, title, ylabel in [
        (axes[0], rmse, "RMSE (↓更好)", "RMSE"),
        (axes[1], mae, "MAE (↓更好)", "MAE"),
        (axes[2], r2, "R2 (↑更好)", "R2"),
    ]:
        bars = ax.bar(names, values, color=[colors.get(n, "#64748B") for n in names])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("GeoNLA vs FlatMLP 验证集指标", fontsize=13)
    out_b = fig_dir / "fig7_ablation_metrics.png"
    fig.tight_layout()
    fig.savefig(out_b, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_b)

    # 图C：分维 R2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(config.OUTPUT_NAMES))
    width = 0.35
    for i, name in enumerate(names):
        r2_dim = [
            0.0 if v is None else float(v)
            for v in results[name]["metrics"]["r2_per_dim"]
        ]
        ax.bar(
            x + (i - 0.5) * width,
            r2_dim,
            width=width,
            label=name,
            color=colors.get(name, "#64748B"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(config.OUTPUT_NAMES, rotation=20, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("分维 R2：人类活动各输出")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="#94A3B8", lw=1)
    out_c = fig_dir / "fig8_ablation_r2_per_dim.png"
    fig.tight_layout()
    fig.savefig(out_c, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(out_c)

    return paths


def generate_all_visualizations(
    checkpoint_path: Path | None = None,
    data_path: Path | None = None,
) -> list[Path]:
    print("=" * 50)
    print("生成 GeoNLA 可视化图表")
    print("=" * 50)

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    model, ckpt, X, y, time_factor, socio_factor = _load_bundle(checkpoint_path, data_path)

    # weight heatmaps 仅对 GeoNLA 有意义
    paths = [
        plot_loss_curves(ckpt, config.FIG_DIR),
        plot_model_architecture(config.FIG_DIR),
        plot_emergence_tsne(model, X, time_factor, socio_factor, config.FIG_DIR),
        plot_prediction_comparison(model, X, time_factor, socio_factor, y, config.FIG_DIR),
        plot_temporal_modulation(model, X, config.FIG_DIR),
    ]
    if hasattr(model, "W1") and hasattr(model, "W2"):
        paths.insert(4, plot_weight_heatmaps(model, config.FIG_DIR))

    print("\n已生成:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    generate_all_visualizations()
