# 2024-02-28
# Caroline Haslebacher
# this script applies LineaMapper to all isis4 Lommel-Seeliger corrected input geotiff
# it reprojects first to orthographic projection and then retrieves predictions with LineaMapper

#%%  import
from pathlib import Path
import os
import json
from osgeo import gdal
import numpy as np
from datetime import datetime
import time

#%% define basepath of data
current = os.getcwd()
titaniach = Path(current.split('Caroline')[0]) / 'Caroline'

basepath = titaniach / 'lineament_detection/RegionalMaps_CH_NT_EJL_LP/mapping/data/isis4_all'
# 
geocubes = sorted((basepath / 'regmaps').glob('*.cub'))

# equipath = basepath / 'polygons/for_analysis/Equirectangular'
# mercatorpath = basepath / 'geocubes/for_analysis/Mercator'
# orthopath = basepath / 'geocubes/for_analysis/Orthographic'

# define where to save sinusoidal projection files
ortho_prj_path = basepath / 'orthographic_projection'
os.makedirs(ortho_prj_path, exist_ok=True)


#%% 1st step: reproject to orthographic

def coordm_to_lon(coordm): # coordinate in metres from qgis
    radius = 1560800 # metres (corrected on 2023-12-06)
    
    if coordm < 0:
        lon = coordm/(np.pi*radius) * 180
        # For longitudes to the left of the 0 median in qgis:Take absolute value and add 180°​
        lon = abs(lon)+180
    else:
        lon = coordm/(np.pi*radius) * 180
        # then, in QGIS, it goes from the median to the right instead of to the left, like in the usgs map
        lon = 180-lon
    return lon

def coordm_to_lat(coordm):
    radius = 1560800 # metres (corrected on 2023-12-06)
    lat = coordm/(np.pi*radius)*180
    return lat

#%%

for geosin in geocubes:
    # print(geosin)
    dataset = gdal.Open(geosin.as_posix(), gdal.GA_ReadOnly)
    ulx, xres, xskew, uly, yskew, yres  = dataset.GetGeoTransform()
    # print(ulx)
    # get central coordinates of dataset
    # central_merid = '{:.3f}'.format(360-coordm_to_lon(ulx + (0.5*dataset.RasterXSize * xres))) # x is longitude
    centre_lat = '{:.4f}'.format(coordm_to_lat(uly + (0.5*dataset.RasterYSize * yres))) # x is longitude
    centre_lon = '{:.4f}'.format(360-coordm_to_lon(ulx + (0.5*dataset.RasterXSize * xres))) # x is longitude
    # print(centre_lat)
    # print(centre_lon)
    # define projection file (copied from S:\Groups\PIG\Caroline\lineament_detection\galileo_manual_segmentation\data\polygons\for_analysis\Sinusoidal\Sinusoidal_EUROPA_0.prj)
    orthographic = f'PROJCS["Orthographic_EUROPA_template_2",GEOGCS["New Geographic Coordinate System",DATUM["<Custom>",SPHEROID["<Custom>",1560800.0,0.0]],PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Orthographic"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Longitude_Of_Center",{centre_lon}],PARAMETER["Latitude_Of_Center",{centre_lat}],UNIT["Meter",1.0]]'
    # write to file
    print(orthographic)
    with open(ortho_prj_path.joinpath("{}.prj".format(geosin.stem)), "w") as prf_file:
        prf_file.write(orthographic)
    # close dataset by deleting variable in python
    del dataset

# %%

dt_string = datetime.now().strftime("%Y_%m_%d_%H")
# NOTE: the savepath does not yet need to exist
basepath_lineamapper = titaniach / 'lineament_detection/RegionalMaps_CH_NT_EJL_LP/mapping'
savepath = basepath_lineamapper / 'output/LineaMapper_output' / dt_string
# for shapefiles:
os.makedirs(Path(savepath) / 'shape_files', exist_ok=True)

equipath = basepath_lineamapper / 'data/isis4_all/equirectangular_projection_file' # S:\Groups\PIG\Caroline\lineament_detection\RegionalMaps_CH_NT_EJL_LP\mapping\data\isis4_all\equirectangular_projection_file
equifile = 'Equirectangular_EUROPA.prj'

# equirectangular, geotiff
equitiff_path = basepath / 'Equirectangular_tif'
os.makedirs(equitiff_path, exist_ok=True)

orthoind_path = basepath / 'Orthographic_tif'
os.makedirs(orthoind_path, exist_ok=True)

# NOTE: I could also try out other projections apart from equirectangular!

#%%

# start full time
full_time_start = time.time()

#%% reproject and directly get predictions

for geocub in geocubes:
    # reproject to mercator and sinusoidal
    # example command: gdalwarp -t_srs ./Mercator/Mercator_EUROPA_0.prj input.tif output.tif

    # gdal: first optional step: cubes to tiff:
    # in: geocub
    # out: equitiff_path
    # gdal_translate -of GTiff input.cub output.tif
    # this is just for me to visualize and load into ArcGIS
    command = 'gdal_translate -of GTiff {} {}'.format(geocub, equitiff_path.joinpath("{}.tif".format(geocub.stem)))
    print(command)
    os.system(command)   

    # orthographic
    command = 'gdalwarp -s_srs {} -t_srs {} {} {}'.format(equipath.joinpath(equifile).as_posix(), ortho_prj_path.joinpath("{}.prj".format(geocub.stem)).as_posix(), geocub, orthoind_path.joinpath(geocub.stem + '.tif').as_posix())
    print(command)
    os.system(command)   

    # be careful, here, we have to take the path where the reprojected orthographic files live as -geofile
    tf = geocub.stem
    print(tf)
    command = 'python LineaMapper_v2_to_img.py --geofile=' + str(orthoind_path.joinpath(tf + '.tif')) + ' --savedir=' + str(savepath) + ' --class_scores 0.5 0.5 0.5 0.5 --multiplication_factor=25 --azimuth_diff_range=25 --cut_size=3000' # --class_scores 0.65 0.3 0.55 0.7 --multiplication_factor=8
    print(command)
    os.system(command)



# measure time and simply print
timesum = time.time() - full_time_start
print('this script took {:.2f} seconds to execute. Makes {:.2f} hours.'.format(timesum, timesum/3600))

# simple test: 
# python LineaMapper_to_img.py --geofile=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/E6ESCRATER01_GalileoSSI_Equi-cog.tif --savedir=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/LineaMapper_output/tests

#%%
######## convert to shapefiles
# problem is that I do not have the exact filename, so I retrieve it simply afterwards
# make shape_file folder
os.makedirs(Path(savepath) / 'shape_files', exist_ok=True)

geojfiles = sorted((Path(savepath) / 'json_files').glob('*.geojson'))
for geoi, geojfile in enumerate(geojfiles):
    # convert to shapefile as well
    command = 'ogr2ogr -nlt POLYGON -makevalid -skipfailures {} {}'.format(savepath + '/shape_files/' + geojfile.stem + '.shp', savepath + '/json_files/' + geojfile.stem + '.geojson')
    print(command)
    os.system(command)
    # append to shapefile https://www.northrivergeographic.com/ogr2ogr-merge-shapefiles/ 
    if geoi == 0:
        # first, initiate a shapefile
        command = 'ogr2ogr -f "ESRI Shapefile" -makevalid -skipfailures {} {}'.format(savepath + '/shape_files/merged_' + tf + '.shp', savepath + '/shape_files/' + geojfile.stem + '.shp')
        print(command)
        os.system(command)
    else: # append
        command = 'ogr2ogr -f "ESRI Shapefile" -makevalid -skipfailures -update -append {} {}'.format(savepath + '/shape_files/merged_' + tf + '.shp', savepath + '/shape_files/' + geojfile.stem + '.shp')
        print(command)
        os.system(command)            
    # ogr2ogr -f ‘ESRI Shapefile’ merge.shp station_1.shp
    # Then merge the following files by using:
    # ogr2ogr -f ‘ESRI Shapefile’ -update -append merge.shp station_3.shp -nln merge

# check if it worked for all
geoshpfiles = sorted((Path(savepath) / 'shape_files').glob('*.shp'))

if (len(geoshpfiles) + 1) == len(geojfiles):
    print('ALL GEOJSON FILES WERE CONVERTED TO SHAPEFILES.')
else:
    print('some geojson files lead to errors, it seems.')
#%% 2nd step: retrieve predictions and produce shapefiles as well.




