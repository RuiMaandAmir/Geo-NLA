"""一键运行：消融对比 + GeoNLA 可视化；可选接入临夏栅格。"""

from __future__ import annotations

import argparse

from ablation import run_ablation
from visualize import generate_all_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoNLA Demo")
    parser.add_argument(
        "--source",
        choices=["synthetic", "raster"],
        default="synthetic",
        help="数据源：synthetic=合成向量；raster=临夏栅格网格",
    )
    parser.add_argument(
        "--prepare-rasters",
        action="store_true",
        help="若缺栅格，先生成临夏演示 GeoTIFF",
    )
    args = parser.parse_args()

    if args.source == "raster":
        import config
        from data_download import check_raw_status
        from data_raster import extract_grid_features

        status = check_raw_status()
        if not all(status.values()):
            if args.prepare_rasters:
                from make_demo_rasters import make_demo_rasters

                print("\n生成演示栅格...")
                make_demo_rasters()
            else:
                raise SystemExit(
                    "栅格不完整。可加 --prepare-rasters 生成演示数据，"
                    "或按 data_download.py 指引放入真实 tif。"
                )
        # 强制重提特征，避免旧缓存
        if config.RASTER_FEATURE_NPZ.exists():
            config.RASTER_FEATURE_NPZ.unlink()
        extract_grid_features(save=True)

    run_ablation(source=args.source)
    data_name = "synthetic_data.npz" if args.source == "synthetic" else "raster_data.npz"
    # ablation 已写 {source}_data.npz；可视化默认读 synthetic，这里显式指定
    import config

    generate_all_visualizations(
        checkpoint_path=config.CHECKPOINT_PATH,
        data_path=config.OUTPUT_DIR / f"{args.source}_data.npz",
    )
    print(f"\nDemo 完成（source={args.source}）。结果见 outputs/")


if __name__ == "__main__":
    main()
