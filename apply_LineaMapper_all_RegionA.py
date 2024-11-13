

# run with:
# conda activate geopytorch
# cd to this directory
# python apply_LineaMapper_...

# this script retrieves predictions on region A for the publication Haslebacher et al. (2024/2025)
# for LineaMapper v1.0
#    on 224 geosize
# for LineaMapper v1.1
#     on 224 geosize (tiles)
#     on 112 geosize
# for LineaMapper v2.0
#     on 112 geosize

#%%

import os
import glob
from pathlib import Path
from datetime import datetime
import time


current = os.getcwd()
titaniach = Path(current.split('Caroline')[0]) / 'Caroline'


#%%


# %% The input tiff is always the same (one native, one photometrically corrected)

basepath = titaniach / 'lineament_detection/Reinforcement_Learning_SAM/europa_surface'

source_path = basepath / 'data/geotiffs/region_A/'
tifpaths = sorted(source_path.glob('*.tif'))

dt_string = datetime.now().strftime("%Y_%m_%d")

#%%

def forward_LM(modelname, version, subset, geosize):
    # NOTE: the savepath does not yet need to exist
    savepath = basepath / 'output/LineaMapper_output' / (dt_string + '_RegionA_' + subset)
    # start full time
    full_time_start = time.time()

    for tifffile in tifpaths:
        tf = tifffile.stem
        print(tf)
        command = f'python LineaMapper_{version}_to_img.py --modelname={modelname} --geofile=' + str(source_path.joinpath(tf + '.tif')) + ' --savedir=' + str(savepath) + f' --class_scores 0.5 0.5 0.5 0.5 --cut_size=3000 --multiplication_factor=15 --azimuth_diff_range=25 --del_pxs=100 --geosize={geosize}' 
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
    os.makedirs(Path(savepath) / 'shape_files', exist_ok=True)

    geojfiles = sorted((Path(savepath) / 'json_files').glob('*.geojson'))
    for geojfile in geojfiles:
        # convert to shapefile as well
        command = 'ogr2ogr -nlt POLYGON -skipfailures {} {}'.format((savepath / 'shape_files').joinpath(geojfile.stem + '.shp'), (savepath / 'json_files').joinpath(geojfile.stem + '.geojson'))
        print(command)
        os.system(command)

    # check if it worked for all
    geoshpfiles = sorted((Path(savepath) / 'shape_files').glob('*.shp'))

    if len(geoshpfiles) == len(geojfiles):
        print('ALL GEOJSON FILES WERE CONVERTED TO SHAPEFILES.')
    else:
        raise Warning('some geojson files lead to errors, it seems.')

    return

#%%

# for LineaMapper v1.0
#    on 224 geosize
subset = 'LM1.0_224' # identification string for savepath
geosize = 224
modelname = './ckpts/Mask_R-CNN_pub2_run23_end_model.pt'
version = 'v1' # 'v1' or 'v2'
# main
forward_LM(modelname, version, subset, geosize)

# for LineaMapper v1.1
#     on 224 geosize (tiles)
subset = 'LM1.1_224' # identification string for savepath
geosize = 224
modelname = "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run10_end_model.pt"
version = 'v1' # 'v1' or 'v2'
# main
forward_LM(modelname, version, subset, geosize)
#     on 112 geosize
geosize = 112
subset = 'LM1.1_112'
forward_LM(modelname, version, subset, geosize)


# for LineaMapper v2.0
#     on 112 geosize
# note that sampath and sammodus are the default. "./ckpts/bbox_vit_b_final.pt", 'vit_b'
subset = 'LM2.0_112' # identification string for savepath
geosize = 112
modelname = "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run10_end_model.pt"
version = 'v2' # 'v1' or 'v2'
# main
forward_LM(modelname, version, subset, geosize)

