
# 2024-06-01, Caroline Haslebacher
# run with:
# cd to this directory
# conda activate pipgeosam
# python apply_LineaMapper_...

#%%

import os
import glob
from pathlib import Path
from datetime import datetime
import time


current = os.getcwd()
titaniach = Path(current.split('Caroline')[0]) / 'Caroline'

# %%
basepath = titaniach / 'lineament_detection/Reinforcement_Learning_SAM/europa_surface'

source_path = basepath / 'data/geotiffs/for_preds/'
tifpaths = sorted(source_path.glob('*.tif'))

dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
# NOTE: the savepath does not yet need to exist
savepath = basepath / 'output/LineaMapper_output' / dt_string
os.makedirs(savepath, exist_ok=True)

# NOTE: I could also try out other projections apart from equirectangular!

#%%

# start full time
full_time_start = time.time()

for tifffile in tifpaths:
    tf = tifffile.stem
    print(tf)
    command = 'python LineaMapper_v1_to_img.py --geofile=' + str(source_path.joinpath(tf + '.tif')) + ' --savedir=' + str(savepath) + ' --class_scores 0.5 0.5 0.5 0.5 --cut_size=3000 --multiplication_factor=25 --azimuth_diff_range=25 --del_pxs=100 --geosize=112' 
    print(command)
    os.system(command)

# measure time and simply print
timesum = time.time() - full_time_start
print('this script took {:.2f} seconds to execute. Makes {:.2f} hours.'.format(timesum, timesum/3600))

# simple test: 
# python LineaMapper_to_img.py --geofile=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/E6ESCRATER01_GalileoSSI_Equi-cog.tif --savedir=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/LineaMapper_output/tests

#%% to shapefile
######## convert to shapefiles
# problem is that I do not have the exact filename, so I retrieve it simply afterwards
# make shape_file folder
os.makedirs(Path(savepath) / 'shape_files', exist_ok=True)

geojfiles = sorted((Path(savepath) / 'json_files').glob('*.geojson'))
for geoi, geojfile in enumerate(geojfiles):
    # convert to shapefile as well
    command = 'ogr2ogr -nlt POLYGON -makevalid -skipfailures {} {}'.format((savepath / 'shape_files').joinpath(geojfile.stem + '.shp'), (savepath / 'json_files').joinpath(geojfile.stem + '.geojson'))
    print(command)
    os.system(command)
    # append to shapefile https://www.northrivergeographic.com/ogr2ogr-merge-shapefiles/ 
    if geoi == 0:
        # first, initiate a shapefile
        command = 'ogr2ogr -f "ESRI Shapefile" -makevalid -skipfailures {} {}'.format((savepath / 'shape_files').joinpath('merged_' + tf + '.shp'), (savepath / 'shape_files').joinpath(geojfile.stem + '.shp'))
        print(command)
        os.system(command)
    else: # append
        command = 'ogr2ogr -f "ESRI Shapefile" -makevalid -skipfailures -update -append {} {}'.format((savepath / 'shape_files').joinpath('merged_' + tf + '.shp'), (savepath / 'shape_files').joinpath(geojfile.stem + '.shp'))
        print(command)
        os.system(command)            
    # ogr2ogr -f ‘ESRI Shapefile’ merge.shp station_1.shp
    # Then merge the following files by using:
    # ogr2ogr -f ‘ESRI Shapefile’ -update -append merge.shp station_3.shp -nln merge

# check if it worked for all
geoshpfiles = sorted((savepath / 'shape_files').glob('*.shp'))

if (len(geoshpfiles) + 1) == len(geojfiles):
    print('ALL GEOJSON FILES WERE CONVERTED TO SHAPEFILES.')
else:
    print('some geojson files lead to errors, it seems.')

#%%