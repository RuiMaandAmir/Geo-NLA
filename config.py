"""GeoNLA Demo 超参数与常量配置。"""

from pathlib import Path

# 路径
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_PATH = OUTPUT_DIR / "geonla_model_checkpoint.pth"
FIG_DIR = OUTPUT_DIR / "figures"

# 临夏栅格数据目录
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RASTER_FEATURE_CSV = PROCESSED_DIR / "linxia_grid_features.csv"
RASTER_FEATURE_NPZ = PROCESSED_DIR / "linxia_grid_features.npz"

# 模型维度
INPUT_DIM = 12
HIDDEN_DIM = 32
OUTPUT_DIM = 6

# 训练
SEED = 42
N_SAMPLES = 400
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2
DROPOUT = 0.2

# 特征名称（与合成数据、可视化一致）
INPUT_NAMES = [
    "海拔", "坡度", "起伏度",
    "温度", "降水", "干燥度",
    "有机质", "黏粒", "持水量",
    "NDVI", "NDVI振幅", "草地比例",
]

OUTPUT_NAMES = [
    "耕地比例", "灌溉比例", "建设用地",
    "人口密度", "夜间灯光", "道路密度",
]

# 临夏盆地风格参数（合成数据用，非真实坐标采样）
LINXIA_ELEV_MEAN = 2000.0
LINXIA_ELEV_STD = 350.0
LINXIA_TEMP_MEAN = 7.0
LINXIA_PRECIP_MEAN = 500.0

# 临夏盆地研究区（WGS84）
# bbox: (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max)
LINXIA_BBOX = (102.9, 35.3, 103.5, 35.8)
LINXIA_CRS = "EPSG:4326"
# 约 1km 网格（演示速度与样本量平衡）；真实实验可改 0.0045≈500m
GRID_STEP_DEG = 0.009

# 期望的原始栅格文件名（放到 data/raw/）
RASTER_FILES = {
    "dem": "dem_linxia.tif",
    "temp": "temp_linxia.tif",          # WorldClim bio01
    "precip": "precip_linxia.tif",      # WorldClim bio12
    "soc": "soc_linxia.tif",            # SoilGrids
    "clay": "clay_linxia.tif",
    "ndvi": "ndvi_linxia.tif",
    "ndvi_amp": "ndvi_amp_linxia.tif",
    "landcover": "worldcover_linxia.tif",
    "population": "worldpop_linxia.tif",
    "nightlight": "viirs_linxia.tif",
}
