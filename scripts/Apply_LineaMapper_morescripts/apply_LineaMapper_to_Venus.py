

# 
#%%

import os
import glob
from pathlib import Path
from datetime import datetime
import time


current = os.getcwd()
titaniach = Path(current.split('Caroline')[0]) / 'Caroline'

# %%
dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
# get path to basemap
basepath_data = titaniach / 'lineament_detection/global_maps_for_book/data/for_analysis/geotiffs/Venus/'
tifpaths = sorted(basepath_data.glob('*.tif'))

#%%

# start full time
full_time_start = time.time()

# basepath_data = Path('z:/Groups/PIG/Caroline/lineament_detection/pytorch_maskrcnn/projects/usgs_basemap_azimuths/')
# 'z:\Group\PIG\Caroline\isis\lineament_detection\pytorch_maskrcnn\projects\usgs_basemap_azimuths\'

# quickly prepare data
# convert first to geotiff (gdal automatically uses the .tfw file to write to geotiff)
# command = 'gdal_translate -of GTiff Z:\Groups\PIG\Caroline\lineament_detection\pytorch_maskrcnn\projects\usgs_basemap_azimuths\Europa_Voyager_GalileoSSI_global_mosaic_500m_arcgisexp.tif Z:\Groups\PIG\Caroline\lineament_detection\pytorch_maskrcnn\data\usgs_basemap_azimuths\Europa_Voyager_GalileoSSI_global_mosaic_500m_gdal.tif' # input.tif output.tif
# did not work (installation problems, and usage problems): gdal2tiles --tilesize=273 Europa_Voyager_GalileoSSI_global_mosaic_500m_arcgisexp.tif tiles_273px
# print(command)
# os.system(command)
# NOTE: I've run the above command (without python) on the command line


for tifffile in tifpaths:
    tf = tifffile.stem
    # savepath for individual tif files
    savepath = 'z:/Groups/PIG/Caroline/lineament_detection/global_maps_for_book/output/LineaMapper_Venus/' + dt_string + tf
    os.makedirs(savepath, exist_ok=True)
    # 5° grid, 273px cut size command = 'python LineaMapper_to_img.py --geofile=Z:/Groups/PIG/Caroline/lineament_detection/pytorch_maskrcnn/data/usgs_basemap_azimuths/' + tf + '.tif --savedir=' + savepath + ' --geosize=273 --cut_size=273 --azimuth_diff_range=10 --class_scores 0.65 0.3 0.55 0.7 --multiplication_factor=8'
    # 15° grid, 818px cut size: command = 'python LineaMapper_to_img.py --geofile=Z:/Groups/PIG/Caroline/lineament_detection/pytorch_maskrcnn/data/usgs_basemap_azimuths/' + tf + '.tif --savedir=' + savepath + ' --geosize=273 --cut_size=818 --azimuth_diff_range=10 --class_scores 0.65 0.3 0.55 0.7 --multiplication_factor=8'
    # 10° grid, 546 px cut size. Geosize adapted to 273 to have two images per cut
    command = 'python LineaMapper_v2_to_img.py --geofile=z:/Groups/PIG/Caroline/lineament_detection/global_maps_for_book/data/for_analysis/geotiffs/Venus/' + tf + '.tif --savedir=' + savepath + ' --geosize=100 --cut_size=600 --class_scores 0.5 0.3 0.5 0.3 --azimuth_diff_range=25 --del_pxs=100 --multiplication_factor=25'
    # for lower-res areas: geosize=500
    # command = 'python LineaMapper_to_img.py --geofile=z:/Groups/PIG/Caroline/lineament_detection/global_maps_for_book/data/for_analysis/geotiffs/original_ArcGIS_export_basemap/' + tf + '.tif --savedir=' + savepath + ' --geosize=500 --cut_size=500 --class_scores 0.9 0.4 0.7 0.4 --azimuth_diff_range=5 --multiplication_factor=6'

    print(command)
    os.system(command)

    # measure time and simply print
    timesum = time.time() - full_time_start
    print('this script took {:.2f} seconds to execute. Makes {:.2f} hours.'.format(timesum, timesum/3600))

    # simple test: 
    # python LineaMapper_to_img.py --geofile=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/E6ESCRATER01_GalileoSSI_Equi-cog.tif --savedir=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/LineaMapper_output/tests

    ######## convert to shapefiles
    # problem is that I do not have the exact filename, so I retrieve it simply afterwards
    # make shape_file folder
    print(savepath)
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

    # # lets merge the shapefiles
    # hmm, this only opens the orgmerge.py script...
    # command = 'python ogrmerge.py -single -f "ESRI Shapefile" -o merged.shp {}*.shp'.format(savepath + '/shape_files/') # ogrmerge.py -single -f ‘ESRI Shapefile’ -o merged.shp *.shp
    # print(command)
    # os.system(command)

# %%
