/**
 * GeoNLA — 临夏盆地 GEE 导出脚本
 *
 * 用法：
 * 1. 打开 https://code.earthengine.google.com/
 * 2. 新建脚本，粘贴本文件全部内容
 * 3. 确认已登录，Run
 * 4. 在 Tasks 面板逐个点 Run，导出到 Google Drive 文件夹 "GeoNLA_Linxia"
 * 5. 下载后裁剪/重命名，放入 geonla_demo/data/raw/（文件名见下方映射）
 *
 * 文件名映射（导出后请改名）：
 *   dem_linxia.tif
 *   temp_linxia.tif          <- bio01
 *   precip_linxia.tif        <- bio12
 *   ndvi_linxia.tif
 *   ndvi_amp_linxia.tif
 *   worldcover_linxia.tif
 *   worldpop_linxia.tif
 *   viirs_linxia.tif
 *
 * SoilGrids（soc/clay）GEE 上不一定稳定，建议另从 https://soilgrids.org/ 下载。
 */

// ========== 研究区 ==========
var bbox = ee.Geometry.Rectangle([102.9, 35.3, 103.5, 35.8]); // minx,miny,maxx,maxy
Map.centerObject(bbox, 9);
Map.addLayer(bbox, {color: 'red'}, 'Linxia BBOX');

var DRIVE_FOLDER = 'GeoNLA_Linxia';
var CRS = 'EPSG:4326';
var SCALE_DEM = 30;
var SCALE_CLIM = 1000;
var SCALE_10M = 10;
var SCALE_100M = 100;
var SCALE_500M = 500;

function exportGeoTiff(image, desc, scale) {
  Export.image.toDrive({
    image: image.clip(bbox),
    description: desc,
    folder: DRIVE_FOLDER,
    fileNamePrefix: desc,
    region: bbox,
    scale: scale,
    crs: CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

// ========== 1. DEM (SRTM 30m) ==========
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('dem');
exportGeoTiff(dem, 'dem_linxia', SCALE_DEM);
Map.addLayer(dem.clip(bbox), {min: 1600, max: 3200, palette: ['#006633', '#E5FFCC', '#662A00']}, 'DEM', false);

// ========== 2. WorldClim bio01 / bio12 ==========
// WorldClim V1 in GEE: WORLDCLIM/V1/BIO
var bioclim = ee.Image('WORLDCLIM/V1/BIO');
var temp = bioclim.select('bio01').multiply(0.1).rename('temp'); // °C
var precip = bioclim.select('bio12').rename('precip');           // mm
exportGeoTiff(temp, 'temp_linxia', SCALE_CLIM);
exportGeoTiff(precip, 'precip_linxia', SCALE_CLIM);

// ========== 3. Sentinel-2 年均 NDVI + 季节振幅 ==========
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(bbox)
  .filterDate('2023-01-01', '2024-01-01')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(function(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    return ndvi.updateMask(ndvi.gt(-0.2).and(ndvi.lt(1.0)))
               .copyProperties(img, ['system:time_start']);
  });

var ndviMean = s2.mean().rename('ndvi');
var ndviAmp = s2.max().subtract(s2.min()).rename('ndvi_amp');
exportGeoTiff(ndviMean, 'ndvi_linxia', SCALE_10M);
exportGeoTiff(ndviAmp, 'ndvi_amp_linxia', SCALE_10M);
Map.addLayer(ndviMean.clip(bbox), {min: 0, max: 0.8, palette: ['#ffffcc', '#006837']}, 'NDVI', false);

// ========== 4. ESA WorldCover 2021 ==========
var worldcover = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('landcover');
exportGeoTiff(worldcover, 'worldcover_linxia', SCALE_10M);
Map.addLayer(worldcover.clip(bbox), {}, 'WorldCover', false);

// ========== 5. WorldPop 人口密度 ==========
var pop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filter(ee.Filter.eq('country', 'CHN'))
  .filterDate('2020-01-01', '2021-01-01')
  .mosaic()
  .rename('population');
exportGeoTiff(pop, 'worldpop_linxia', SCALE_100M);

// ========== 6. VIIRS 夜间灯光 (2022 年均) ==========
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate('2022-01-01', '2023-01-01')
  .select('avg_rad')
  .mean()
  .rename('nightlight');
exportGeoTiff(viirs, 'viirs_linxia', SCALE_500M);

print('Tasks 已提交到导出队列。请到 Tasks 面板逐个 Run。');
print('完成后下载到本地，按 README 映射改名放入 data/raw/');
print('然后运行: python data_raster.py && python run_demo.py --source raster');
