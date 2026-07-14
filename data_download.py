"""
临夏盆地真实栅格获取指引与状态检查。

推荐流程（GEE）：
  1) 在 https://code.earthengine.google.com/ 运行 gee_export_linxia.js
  2) Tasks 导出到 Drive 文件夹 GeoNLA_Linxia，下载到本地
  3) python ingest_gee_downloads.py --src "你的下载目录"
  4) SoilGrids 的 soc/clay 另从 https://soilgrids.org/ 下载放入 data/raw/
     （若暂时没有，可保留 make_demo_rasters 生成的同名文件占位）
  5) python data_raster.py
  6) python run_demo.py --source raster

演示占位：
  python make_demo_rasters.py
"""

from __future__ import annotations

import config


DOWNLOAD_GUIDE = {
    "dem": {
        "file": config.RASTER_FILES["dem"],
        "source": "SRTM 30m",
        "url": "https://dwtkns.com/srtm30m/",
        "note": "下载 N35E102 / N35E103，裁剪到临夏 bbox",
    },
    "temp": {
        "file": config.RASTER_FILES["temp"],
        "source": "WorldClim bio01",
        "url": "https://www.worldclim.org/data/worldclim21.html",
        "note": "30s bioclimatic variables",
    },
    "precip": {
        "file": config.RASTER_FILES["precip"],
        "source": "WorldClim bio12",
        "url": "https://www.worldclim.org/data/worldclim21.html",
        "note": "年降水",
    },
    "soc": {
        "file": config.RASTER_FILES["soc"],
        "source": "SoilGrids SOC",
        "url": "https://soilgrids.org/",
        "note": "soil organic carbon",
    },
    "clay": {
        "file": config.RASTER_FILES["clay"],
        "source": "SoilGrids Clay",
        "url": "https://soilgrids.org/",
        "note": "clay content",
    },
    "ndvi": {
        "file": config.RASTER_FILES["ndvi"],
        "source": "Sentinel-2 / Landsat NDVI",
        "url": "https://earthengine.google.com/",
        "note": "可用 GEE 导出临夏年均 NDVI",
    },
    "ndvi_amp": {
        "file": config.RASTER_FILES["ndvi_amp"],
        "source": "NDVI 季节振幅",
        "url": "https://earthengine.google.com/",
        "note": "年最大-最小 NDVI",
    },
    "landcover": {
        "file": config.RASTER_FILES["landcover"],
        "source": "ESA WorldCover",
        "url": "https://esa-worldcover.org/en/data-access",
        "note": "类别编码保持 10/30/40/50",
    },
    "population": {
        "file": config.RASTER_FILES["population"],
        "source": "WorldPop",
        "url": "https://www.worldpop.org/data",
        "note": "人口密度",
    },
    "nightlight": {
        "file": config.RASTER_FILES["nightlight"],
        "source": "VIIRS Nighttime Lights",
        "url": "https://eogdata.mines.edu/products/vnl/",
        "note": "夜间灯光",
    },
}


def check_raw_status() -> dict[str, bool]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    status = {}
    print(f"临夏 bbox: {config.LINXIA_BBOX}")
    print(f"原始目录: {config.RAW_DIR}\n")
    for key, meta in DOWNLOAD_GUIDE.items():
        path = config.RAW_DIR / meta["file"]
        ok = path.exists()
        status[key] = ok
        mark = "OK" if ok else "缺"
        print(f"[{mark}] {meta['file']}")
        if not ok:
            print(f"     来源: {meta['source']}")
            print(f"     链接: {meta['url']}")
            print(f"     备注: {meta['note']}")
    n_ok = sum(status.values())
    print(f"\n进度: {n_ok}/{len(status)}")
    if n_ok < len(status):
        print("提示: 可先 python make_demo_rasters.py 生成演示栅格打通流程。")
    return status


if __name__ == "__main__":
    check_raw_status()
