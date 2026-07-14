"""
临夏盆地栅格 → 500m 网格特征矩阵。

优先读取 data/raw/*.tif；若缺失则提示先运行 make_demo_rasters.py
或放入真实裁剪栅格。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config
from data import GeoNLADataset

try:
    import rasterio
    from rasterio.windows import from_bounds
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要安装 rasterio：pip install rasterio") from exc


def _require_rasters() -> dict[str, Path]:
    paths = {}
    missing = []
    for key, fname in config.RASTER_FILES.items():
        p = config.RAW_DIR / fname
        if p.exists():
            paths[key] = p
        else:
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            "缺少栅格文件:\n  - "
            + "\n  - ".join(missing)
            + f"\n请先运行: python make_demo_rasters.py\n"
            f"或将真实临夏裁剪 GeoTIFF 放到: {config.RAW_DIR}"
        )
    return paths


def _grid_boxes(
    bbox: tuple[float, float, float, float] = config.LINXIA_BBOX,
    step: float = config.GRID_STEP_DEG,
) -> list[tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = bbox
    boxes = []
    lon = minx
    while lon < maxx - 1e-12:
        lat = miny
        while lat < maxy - 1e-12:
            boxes.append((lon, lat, min(lon + step, maxx), min(lat + step, maxy)))
            lat += step
        lon += step
    return boxes


def _read_window(src, bbox: tuple[float, float, float, float]) -> np.ndarray:
    window = from_bounds(*bbox, transform=src.transform)
    data = src.read(1, window=window, boundless=True)
    arr = data.astype(np.float64)
    nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr


def _window_mean(src, bbox: tuple[float, float, float, float]) -> float:
    arr = _read_window(src, bbox)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _window_mode_landcover(src, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    """返回耕地/草地/建设用地比例（WorldCover 编码）。"""
    data = _read_window(src, bbox).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"crop": np.nan, "grass": np.nan, "built": np.nan}
    crop = float(np.mean(data == 40))
    grass = float(np.mean(data == 30))
    built = float(np.mean(data == 50))
    return {"crop": crop, "grass": grass, "built": built}


def _estimate_slope_relief(src, bbox) -> tuple[float, float]:
    """用已打开的 DEM 数据集估算坡度均值与起伏度。"""
    dem = _read_window(src, bbox)
    if dem.shape[0] < 2 or dem.shape[1] < 2 or np.all(np.isnan(dem)):
        return np.nan, np.nan

    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    dx = res_x * 111320 * np.cos(np.deg2rad((bbox[1] + bbox[3]) / 2))
    dy = res_y * 110540
    fill = np.nanmean(dem)
    gy, gx = np.gradient(np.nan_to_num(dem, nan=fill), dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = float(np.nanmax(dem) - np.nanmin(dem))
    return float(np.nanmean(slope_deg)), relief


def extract_grid_features(save: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    从 data/raw 栅格提取网格特征。

    Returns
    -------
    X : [N, 12]
    y : [N, 6]
    coords : [N, 2] (lon_center, lat_center)
    df : 原始未标准化表格
    """
    paths = _require_rasters()
    boxes = _grid_boxes()
    print(f"提取临夏网格特征: {len(boxes)} 个格子, step={config.GRID_STEP_DEG}°")

    rows = []
    with rasterio.open(paths["dem"]) as dem_src, \
         rasterio.open(paths["temp"]) as temp_src, \
         rasterio.open(paths["precip"]) as precip_src, \
         rasterio.open(paths["soc"]) as soc_src, \
         rasterio.open(paths["clay"]) as clay_src, \
         rasterio.open(paths["ndvi"]) as ndvi_src, \
         rasterio.open(paths["ndvi_amp"]) as amp_src, \
         rasterio.open(paths["landcover"]) as lc_src, \
         rasterio.open(paths["population"]) as pop_src, \
         rasterio.open(paths["nightlight"]) as nl_src:

        for i, bbox in enumerate(boxes):
            elev = _window_mean(dem_src, bbox)
            slope, relief = _estimate_slope_relief(dem_src, bbox)
            temp = _window_mean(temp_src, bbox)
            precip = _window_mean(precip_src, bbox)
            soc = _window_mean(soc_src, bbox)
            clay = _window_mean(clay_src, bbox)
            ndvi = _window_mean(ndvi_src, bbox)
            ndvi_amp = _window_mean(amp_src, bbox)
            lc = _window_mode_landcover(lc_src, bbox)
            pop = _window_mean(pop_src, bbox)
            night = _window_mean(nl_src, bbox)

            if any(np.isnan(v) for v in [elev, slope, relief, temp, precip, soc, clay, ndvi, ndvi_amp, pop, night]):
                continue
            if any(np.isnan(v) for v in lc.values()):
                continue

            aridity = (temp + 10) / (precip / 100.0 + 1e-3)
            field_cap = 0.15 * soc + 0.2 * clay + 18.0

            # 标签：由土地覆盖/人口/灯光派生（真实数据时同逻辑）
            crop = lc["crop"]
            built = lc["built"]
            grass = lc["grass"]
            irrig = float(np.clip(crop * (0.4 + 0.4 * min(aridity, 3) / 3), 0, 1))
            pop_n = float(np.clip(pop / 2000.0, 0, 1))
            night_n = float(np.clip(night / 40.0, 0, 1))
            road = float(np.clip(0.6 * built + 0.3 * night_n + 0.1 * pop_n, 0, 1))

            lon_c = 0.5 * (bbox[0] + bbox[2])
            lat_c = 0.5 * (bbox[1] + bbox[3])
            rows.append(
                {
                    "lon": lon_c,
                    "lat": lat_c,
                    "海拔": elev,
                    "坡度": slope,
                    "起伏度": relief,
                    "温度": temp,
                    "降水": precip,
                    "干燥度": aridity,
                    "有机质": soc,
                    "黏粒": clay,
                    "持水量": field_cap,
                    "NDVI": ndvi,
                    "NDVI振幅": ndvi_amp,
                    "草地比例": grass,
                    "耕地比例": crop,
                    "灌溉比例": irrig,
                    "建设用地": built,
                    "人口密度": pop_n,
                    "夜间灯光": night_n,
                    "道路密度": road,
                }
            )
            if (i + 1) % 200 == 0:
                print(f"  已处理 {i + 1}/{len(boxes)} ...")

    if not rows:
        raise RuntimeError("未能提取任何有效网格，请检查栅格范围与 nodata。")

    df = pd.DataFrame(rows)
    X_raw = df[config.INPUT_NAMES].to_numpy(dtype=np.float64)
    y = df[config.OUTPUT_NAMES].to_numpy(dtype=np.float64)
    coords = df[["lon", "lat"]].to_numpy(dtype=np.float64)

    # Z-score 标准化输入
    X = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
    y = np.clip(y, 0.0, 1.0)

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RASTER_FEATURE_CSV, index=False, encoding="utf-8-sig")
        np.savez(
            config.RASTER_FEATURE_NPZ,
            X=X,
            y=y,
            X_raw=X_raw,
            coords=coords,
            input_names=np.array(config.INPUT_NAMES),
            output_names=np.array(config.OUTPUT_NAMES),
        )
        print(f"已保存: {config.RASTER_FEATURE_CSV}")
        print(f"已保存: {config.RASTER_FEATURE_NPZ}")
        print(f"有效网格: {len(df)}")

    return X, y, coords, df


def load_linxia_raster_dataset(seed: int = config.SEED) -> GeoNLADataset:
    """
    加载栅格特征为 GeoNLADataset。
    若已有 processed npz 则直接读；否则现场提取。
    """
    if config.RASTER_FEATURE_NPZ.exists():
        data = np.load(config.RASTER_FEATURE_NPZ, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        coords = data["coords"]
        print(f"从缓存加载栅格特征: {config.RASTER_FEATURE_NPZ}  N={len(X)}")
    else:
        X, y, coords, _ = extract_grid_features(save=True)

    rng = np.random.default_rng(seed)
    n = len(X)
    month = rng.integers(1, 13, size=n)
    time_factor = np.sin(2 * np.pi * month / 12.0).astype(np.float64)[:, None]
    # 社会经济代理：夜间灯光+人口密度的粗略综合（若 y 已含，用其派生）
    socio_factor = np.clip(
        0.5 * y[:, 3:4] + 0.5 * y[:, 4:5] + rng.normal(0, 0.05, size=(n, 1)),
        0.05,
        0.95,
    )

    return GeoNLADataset(
        X=X.astype(np.float64),
        y=y.astype(np.float64),
        time_factor=time_factor,
        socio_factor=socio_factor,
        month=month.astype(np.int64),
        coords=coords.astype(np.float64),
    )


if __name__ == "__main__":
    extract_grid_features(save=True)
