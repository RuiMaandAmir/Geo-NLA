"""
把 GEE / 手动下载的 GeoTIFF 整理进 data/raw/。

用法示例：
  python ingest_gee_downloads.py --src "D:/Downloads/GeoNLA_Linxia"
  python ingest_gee_downloads.py --src "D:/Downloads/GeoNLA_Linxia" --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config

# 导出文件名前缀 / 常见别名 → 项目标准文件名
ALIASES = {
    "dem_linxia": "dem_linxia.tif",
    "temp_linxia": "temp_linxia.tif",
    "precip_linxia": "precip_linxia.tif",
    "ndvi_linxia": "ndvi_linxia.tif",
    "ndvi_amp_linxia": "ndvi_amp_linxia.tif",
    "worldcover_linxia": "worldcover_linxia.tif",
    "worldpop_linxia": "worldpop_linxia.tif",
    "viirs_linxia": "viirs_linxia.tif",
    # SoilGrids 手动下载时可用这些别名
    "soc_linxia": "soc_linxia.tif",
    "clay_linxia": "clay_linxia.tif",
}


def _match_target(stem: str) -> str | None:
    s = stem.lower()
    for key, fname in ALIASES.items():
        if s == key or s.startswith(key):
            return fname
    return None


def ingest(src: Path, dry_run: bool = False) -> dict[str, str]:
    if not src.exists():
        raise FileNotFoundError(src)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    copied = {}
    tifs = list(src.rglob("*.tif")) + list(src.rglob("*.tiff"))
    if not tifs:
        print(f"未找到 tif: {src}")
        return copied

    for path in tifs:
        target_name = _match_target(path.stem)
        if target_name is None:
            print(f"  跳过（无法识别）: {path.name}")
            continue
        dest = config.RAW_DIR / target_name
        print(f"  {path.name} -> {dest}")
        if not dry_run:
            shutil.copy2(path, dest)
        copied[target_name] = str(path)

    print(f"\n完成: {len(copied)} 个文件 → {config.RAW_DIR}")
    print("若仍缺 soc/clay，请从 SoilGrids 下载后放入同目录。")
    print("然后: python data_raster.py")
    print("      python run_demo.py --source raster")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="整理临夏栅格到 data/raw")
    parser.add_argument("--src", required=True, help="GEE Drive 下载文件夹路径")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    ingest(Path(args.src), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
