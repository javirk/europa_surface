# Caroline Haslebacher
# 2023-01-25
# this script simply reads in the metricsdict json file save directly by cocoeval.py
# (C:\Users\ch20s351\Anaconda3\envs\pytorch\Lib\site-packages\pycocotools\cocoeval.py)
# converts it to a dataframe for a nice table view and saves it as a csv file that can be read in directly with Latex

#%%
# import

import json
import pandas as pd

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


from pathlib import Path

#%% 

# coloring of the table
# for precision, we use a colorbar ranging from 0 to 50
# for recall, we use a colorbar from 0 to 100

# viridis = cm.get_cmap('viridis', 256)
# colors = viridis(np.linspace(0, 1, 128))
# newcmp = ListedColormap(colors)
# plot_examples([viridis, newcmp])

def my_to_hex(c, keep_alpha=False):
    # '''
    # CH: I adjusted the original matplotlib function
    # *from matplotlib.colors import to_hex*
    # (C:\Users\ch20s351\Anaconda3\envs\pytorch\Lib\site-packages\matplotlib\colors.py )
    # because I did not wanted to have the # in the string, for Latex.
    # ### CH END

    # Convert *c* to a hex color, no # in the string.

    # Parameters
    # ----------
    # c : :doc:`color </tutorials/colors/colors>` or `numpy.ma.masked`

    # keep_alpha: bool, default: False
    #   If False, use the ``#rrggbb`` format, otherwise use ``#rrggbbaa``.

    # Returns
    # -------
    # str
    #   ``#rrggbb`` or ``#rrggbbaa`` hex color string
    # '''

    
    # c = to_rgba(c)
    if not keep_alpha:
        c = c[:3]
    return "".join(format(int(round(val * 255)), "02x") for val in c)

def calc_diff(new, old, fac=1):
    '''
    simply calculate the difference
    '''
    return (new-old) * fac

def get_sign(y):
    '''
    simply returns a + for positive values
    
    We return an empty string for negative values (since the minus is already placed with the negative number)
    '''

    if y > 0:
        return '+'
    else:
        return ''

def prep_df(jsonpath, jsonfilename, short=False):
    df = pd.read_json(jsonpath.joinpath(jsonfilename))

    # here, we make sure that if the category 'empty' is also in the results, we get rid of it.
    if len(df.index) == 6:
        # remove first index and shift, since then, we have a category 'empty'
        df= df.set_index(np.arange(0, 6))  # set the index (otherwise it is ill-defined and I cannot drop)
        df = df.drop(0) # drop the first row (by using its index 0)
        df = df.set_index(np.array(['0','1','2','3','total'])) # set the index correctly (four labels and a total)
        # also, we need to recompute the mean: (because otherwise, it is taking the empty category into account as well)
        df.loc['total'] = df.mean()

    # create dictionary to convert 'categories' 0-3 to strings of class names
    if short:
        # or shorter:
        categdict = {'0': 'b', '1': 'dr', '2': 'rc', '3': 'ul', 'total': 'mean'}
    else:
        categdict = {'0': 'bands', '1': 'double ridges', '2': 'ridge complexes', '3': 'undiff. lineae', 'total': 'mean'}

    # convert all to percent
    df = df*100

    # cast index to category name
    # TODO: choose between short and long names (depending what fits better)
    df['unit'] = [categdict[x] for x in df.index]

    # construct dataframe from 'usable' columns
    # all columns: Index(['AP_0.50:0.95_all', 'AP_0.50_all', 'AP_0.75_all', 'AP_0.50:0.95_small',
    #        'AP_0.50:0.95_medium', 'AP_0.50:0.95_large', 'AR_0.50:0.95_all',
    #        'AR_0.50:0.95_small', 'AR_0.50:0.95_medium', 'AR_0.50:0.95_large',
    #        'AP_0.35_all', 'AR_0.35_all', 'AR_0.50_all', 'AR_0.75_all'],
    #       dtype='object')
    df_out = df[[
        'unit',
        #### PRECISION
        'AP_0.35_all', 'AP_0.50_all', 'AP_0.50:0.95_all', 
        'AP_0.50:0.95_small', 'AP_0.50:0.95_medium', 'AP_0.50:0.95_large',
        
        #### RECALL
        'AR_0.35_all', 'AR_0.50_all', 'AR_0.50:0.95_all', 
        'AR_0.50:0.95_small', 'AR_0.50:0.95_medium', 'AR_0.50:0.95_large'
                ]]

    # give 'nice' column names in Latex notation
    df_out.columns = ['unit' ,'AP$_{0.35}$', 'AP$_{0.50}$', 'AP',  #_{0.50:0.95} 
                    'AP$_S$', 'AP$_M$', 'AP$_L$',
                    'AR$_{0.35}$', 'AR$_{0.50}$', 'AR', # _{0.50:0.95}
                    'AR$_S$', 'AR$_M$', 'AR$_L$',
                    ]

    return df_out
    
def metricsdict_to_table(jsonfilename, jsonpath, color=True, short=False, comparison={}):
    '''
    to test:
    jsonfilename = '2024_05_02_16_00_22_metricsdict_bbox_v1.1_regiomaps112.json'
    comparison = {'path': '2024_05_03_17_14_28_metricsdict_bbox_v0_regiomaps112.json'}
    '''
    # # Opening JSON file
    # with open(jsonpath.joinpath(jsonfilename)) as json_file:
    #     metricsdict = json.load(json_file)

    # prepare pandas dataframe
    df_out = prep_df(jsonpath, jsonfilename)

    if 'path' in comparison.keys():
        # add a superscript of the absolute difference (increase or decrease)
        # like in \\titania.unibe.ch\Space\Groups\PIG\Caroline\lineament_detection\RegionalMaps_CH_NT_EJL_LP\mapping\code\17ESREGMAP_LineaMapperv1_1\Precision_recall.py 

        # first, prepare dataframe of the comparable data
        df_comp = prep_df(comparison['path'], comparison['name'])

        # calculate difference and write to new dataframe (copied)
        #example: calc_delta(33, 18, fac=100) gives 83.33333
        df_withdiff = df_out.copy()
        # df_withdiff = pd.DataFrame(columns=['bbox_precision', 'bbox_recall', 'masks_precision', 'masks_recall'])
        # df_withdiff.index = ['bands', 'double ridges', 'ridge complexes', 'undiff. lineae', 'average']

        for colname in df_out.columns:
            if colname == 'unit':
                continue
            new = np.array(df_out[colname])
            old = np.array(df_comp[colname])

            # df_withdiff.loc[:,colname] =  df.loc[:,colname].applymap(lambda x: "{:.1f}^{{{}}}".format(x, calc_diff(x, x, fac=100)))

            df_withdiff[colname] =  ["{:.1f} $^{{{}{:.1f}}}$ ".format(x, get_sign(y), y) for (x, y) in zip(new, calc_diff(new, old, fac=1))]

        df_out = df_withdiff

    if color:
        # add colors as string
        # e.g.:
        #\cellcolor[HTML]{ffffe5}{47}
        prec_cbar = np.array([my_to_hex(plt.cm.YlGn(i / 120)) for i in range(120)])
        rec_cbar = np.array([my_to_hex(plt.cm.BuGn(i / 120)) for i in range(120)])
        # example usage: prec_cbar[df_out.iloc[0,1].astype(int)]
        # precision part
        # NOTE: int(x.split('.')[0]) transforms a string from '75.5 ^{+6.6}' to '75', which then goes to 75 by taking int('75')
        if 'path' in comparison.keys():
            df_out.loc[:,'AP$_{0.35}$':'AP$_L$'] =  df_out.loc[:,'AP$_{0.35}$':'AP$_L$'].applymap(lambda x: "\\cellcolor[HTML]{{{}}}{{{}}}".format(prec_cbar[int(x.split('.')[0])], x))
            # recall part
            df_out.loc[:,'AR$_{0.35}$':] =  df_out.loc[:,'AR$_{0.35}$':].applymap(lambda x: "\\cellcolor[HTML]{{{}}}{{{}}}".format(rec_cbar[int(x.split('.')[0])], x))
            # testing string output: "\\cellcolor[HTML]{{{}}}{{{}}}".format(prec_cbar[int(22.5)], 22.5) --> printed as \cellcolor[HTML]{92d183}{22.5}
            # save as csv
            csv_path = jsonpath.joinpath(jsonfilename.split('.')[0] + '_comparison.csv')
        else: # no split needed
            df_out.loc[:,'AP$_{0.35}$':'AP$_L$'] =  df_out.loc[:,'AP$_{0.35}$':'AP$_L$'].applymap(lambda x: "\\cellcolor[HTML]{{{}}}{{{}}}".format(prec_cbar[int(x)], round(x, 1)))
            # recall part
            df_out.loc[:,'AR$_{0.35}$':] =  df_out.loc[:,'AR$_{0.35}$':].applymap(lambda x: "\\cellcolor[HTML]{{{}}}{{{}}}".format(rec_cbar[int(x)], round(x, 1)))
            # save as csv
            csv_path = jsonpath.joinpath(jsonfilename.split('.')[0] + '.csv')
    else:
        csv_path = jsonpath.joinpath(jsonfilename.split('.')[0] + '.csv')

    os.makedirs(csv_path.parents[1], exist_ok=True)
    df_out.to_csv(csv_path, index=False, float_format='%.1f')

    return


def csv_combine_mean_rows(csv_list, csvpath, row_names=[]):
    '''
    takes the 'mean' row from each csv in the list and combines these, with corresponding row names for each mean,
    to a new csv for LaTex
    for debugging:
    row_names=[]
    '''
    means = []
    for ci, csv in enumerate(csv_list):
        df = pd.read_csv(csv)
        mean_row = df.iloc[[4]].copy()
        # change the 'unit' to the specified row name
        if len(row_names) > 0:
            mean_row['unit'] = row_names[ci]
        else:
            mean_row['unit'] = ' '.join(csv.stem.split('_')[8:])
        means.append(mean_row)

    # combine mean rows to a new csv
    df_means = pd.concat(means)
    # save to new csv file
    df_means.to_csv(csvpath)

    return


# %%

if __name__ == '__main__':
    # read in json
    # in folder 'metrics'

    # get all files from subfolders of '.metrics'
    metricspath = sorted(Path('./metrics').glob('*/*.json'))

    for metricp in metricspath:
        jsonpath = metricp.parents[0]
        indfile = metricp.name
        # function call:
        metricsdict_to_table(indfile, jsonpath)

    # and compare to 'old' 100% file

    # bounding box
    # v2.0 on Regiomaps 112
    jsonpath = Path('./metrics/2024_06_04_11_05_Regiomaps_112x112_rcnnSAM/')
    jsonfilename = '2024_06_15_10_54_26_metricsdict_bbox.json'
    # compare with v1.0 on regiomaps 112x112
    comparison = {'path': Path('./metrics/comparison_to_v0/'), 'name': '2024_06_15_15_51_04_metricsdict_bbox_pub1_run23_onRegiomaps112.json'} # '2024_05_03_17_14_28_metricsdict_bbox_v0_regiomaps112.json'}

    # NOTE: this overwrites files generated with the previous cell!
    metricsdict_to_table(jsonfilename, jsonpath, comparison=comparison) 

    # mask
    # v2.0 on Regiomaps 112
    jsonpath = Path('./metrics/2024_06_04_11_05_Regiomaps_112x112_rcnnSAM/')
    jsonfilename = '2024_06_15_10_54_26_metricsdict_segm.json'
    # compare with v1.0 on regiomaps 112x112
    comparison = {'path': Path('./metrics/comparison_to_v0/'), 'name': '2024_06_15_15_51_04_metricsdict_segm_pub1_run23_onRegiomaps112.json'}

    # NOTE: this overwrites files generated with the previous cell!
    metricsdict_to_table(jsonfilename, jsonpath, comparison=comparison) 

    # v1.1 on Regiomaps 112
    jsonpath = Path('./metrics/run10_v1.2_on_regiomaps112/')
    jsonfilename = '2024_06_15_15_50_39_metricsdict_segm_run10_onRegmaps112.json'
    # compare with v1.0 on regiomaps 112x112
    comparison = {'path': Path('./metrics/comparison_to_v0/'), 'name': '2024_06_15_15_51_04_metricsdict_segm_pub1_run23_onRegiomaps112.json'}

    # NOTE: this overwrites files generated with the previous cell!
    metricsdict_to_table(jsonfilename, jsonpath, comparison=comparison) 


    # Regiomaps 112, different model comparison:
    # get all files from subfolders of '.metrics'
    metricspath = sorted(Path('./metrics/2024_06_04_11_05_Regiomaps_112x112_rcnnSAM/additional').glob('*.json'))
    for metricp in metricspath:
        jsonpath = metricp.parents[0]
        indfile = metricp.name
        if 'bbox' in indfile:
            # bbox comparison file
            comparison = {'path': Path('./metrics/comparison_to_v0/'), 'name': '2024_06_15_15_51_04_metricsdict_bbox_pub1_run23_onRegiomaps112.json'}
        elif 'segm' in indfile:
            # mask comparison file
            comparison = {'path': Path('./metrics/comparison_to_v0/'), 'name': '2024_06_15_15_51_04_metricsdict_segm_pub1_run23_onRegiomaps112.json'}

        # function call:
        metricsdict_to_table(indfile, jsonpath, comparison=comparison)

    # only 'segm' csv files
    csv_list = sorted(Path('./metrics/2024_06_04_11_05_Regiomaps_112x112_rcnnSAM/for_mean_segm_comparison').glob('*segm*.csv'))
    csvpath = Path('./metrics/2024_06_04_11_05_Regiomaps_112x112_rcnnSAM/for_mean_segm_comparison').joinpath('combined_model_comparison.csv')
    csv_combine_mean_rows(csv_list, csvpath)


    # photomet
    # get all files from subfolders of '.metrics'
    metricspath = sorted(Path('./metrics/2024_07_04_16_50_photomet').glob('*.json'))

    for metricp in metricspath:
        jsonpath = metricp.parents[0]
        indfile = metricp.name
        # function call:
        metricsdict_to_table(indfile, jsonpath)

# %%
