"""
生成临夏盆地范围的演示 GeoTIFF（地理坐标正确，数值为合成）。

用途：在真实遥感下载完成前，打通「栅格 → 网格特征 → 训练」全流程。
真实数据到位后，用同名文件覆盖 data/raw/ 即可。
"""

from __future__ import annotations

import math

import numpy as np

import config

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "需要安装 rasterio：pip install rasterio\n" + str(exc)
    ) from exc


def _write_tif(path, array: np.ndarray, transform, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=config.LINXIA_CRS,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(array, 1)


def make_demo_rasters(resolution_deg: float = 0.001, seed: int = config.SEED) -> dict:
    """
    按 LINXIA_BBOX 生成演示栅格，分辨率默认约 100m。
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = config.LINXIA_BBOX
    width = int(math.ceil((maxx - minx) / resolution_deg))
    height = int(math.ceil((maxy - miny) / resolution_deg))
    transform = from_origin(minx, maxy, resolution_deg, resolution_deg)

    ys = np.linspace(maxy - resolution_deg / 2, miny + resolution_deg / 2, height)
    xs = np.linspace(minx + resolution_deg / 2, maxx - resolution_deg / 2, width)
    xx, yy = np.meshgrid(xs, ys)

    # 空间场：模拟临夏河谷—山地格局
    valley = np.exp(-((xx - 103.2) ** 2 / 0.02 + (yy - 35.55) ** 2 / 0.015))
    spatial = 0.6 * np.sin((xx - minx) / (maxx - minx) * 2 * np.pi) + 0.4 * valley

    elev = (
        config.LINXIA_ELEV_MEAN
        + config.LINXIA_ELEV_STD * (0.5 - valley)
        + 80 * spatial
        + rng.normal(0, 25, size=(height, width))
    ).astype(np.float32)

    temp = (
        config.LINXIA_TEMP_MEAN
        - 0.006 * (elev - config.LINXIA_ELEV_MEAN)
        + 1.2 * spatial
        + rng.normal(0, 0.4, size=(height, width))
    ).astype(np.float32)

    precip = (
        config.LINXIA_PRECIP_MEAN
        - 0.12 * (elev - config.LINXIA_ELEV_MEAN)
        + 35 * spatial
        + rng.normal(0, 15, size=(height, width))
    ).astype(np.float32)
    precip = np.clip(precip, 150, 900)

    aridity = ((temp + 10) / (precip / 100.0 + 1e-3)).astype(np.float32)
    soc = np.clip(25 - 0.4 * aridity + 0.02 * precip / 10 + rng.normal(0, 1.5, elev.shape), 2, 60).astype(np.float32)
    clay = np.clip(28 + 8 * spatial + rng.normal(0, 3, elev.shape), 5, 55).astype(np.float32)

    ndvi = (1 / (1 + np.exp(-(0.012 * precip - 0.25 * aridity + 0.03 * soc - 2))))
    ndvi = np.clip(ndvi + rng.normal(0, 0.03, elev.shape), 0.05, 0.9).astype(np.float32)
    ndvi_amp = np.clip(0.12 + 0.25 * ndvi + rng.normal(0, 0.02, elev.shape), 0.05, 0.55).astype(np.float32)

    # WorldCover 风格类别：10=林地 30=草地 40=耕地 50=建成 80=水体
    cropland_prob = np.clip(0.55 * valley + 0.2 * ndvi - 0.0004 * (elev - 1800), 0.05, 0.85)
    built_prob = np.clip(0.35 * valley - 0.0003 * (elev - 1900), 0.01, 0.45)
    landcover = np.full(elev.shape, 30, dtype=np.uint8)  # 默认草地
    landcover[cropland_prob > 0.35] = 40
    landcover[built_prob > 0.18] = 50
    landcover[ndvi > 0.65] = 10
    landcover[(xx > 103.15) & (xx < 103.25) & (yy > 35.5) & (yy < 35.6) & (valley > 0.4)] = 50

    population = np.clip(
        800 * built_prob + 120 * cropland_prob + rng.normal(0, 20, elev.shape),
        0,
        5000,
    ).astype(np.float32)
    nightlight = np.clip(
        25 * built_prob + 5 * cropland_prob + rng.normal(0, 0.8, elev.shape),
        0,
        60,
    ).astype(np.float32)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    arrays = {
        "dem": elev,
        "temp": temp,
        "precip": precip,
        "soc": soc,
        "clay": clay,
        "ndvi": ndvi,
        "ndvi_amp": ndvi_amp,
        "landcover": landcover.astype(np.float32),
        "population": population,
        "nightlight": nightlight,
    }
    for key, arr in arrays.items():
        path = config.RAW_DIR / config.RASTER_FILES[key]
        _write_tif(path, arr, transform)
        written[key] = str(path)
        print(f"  写入 {path.name}: shape={arr.shape}")

    meta_path = config.RAW_DIR / "README_rasters.txt"
    meta_path.write_text(
        "临夏盆地演示栅格（合成数值 + 真实 bbox/CRS）。\n"
        "替换为真实数据时，保持文件名与地理范围一致即可。\n"
        f"BBOX={config.LINXIA_BBOX}, CRS={config.LINXIA_CRS}\n",
        encoding="utf-8",
    )
    print(f"演示栅格已写入: {config.RAW_DIR}")
    return written


if __name__ == "__main__":
    make_demo_rasters()
