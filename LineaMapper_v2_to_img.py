# 2022-06-26
# Caroline Haslebacher
# this file generates neural network output (inference)
# you can input one image of arbitrary size
# which gets then tiled up for model inference
# and finally patched together again.

#%%
# import stuff
#%% import

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import box_iou

import os
from pathlib import Path

# import albumentations as A
from datetime import datetime

#%matplotlib inline
import matplotlib.pyplot as plt
from PIL import ImageColor, ImageDraw, ImageFont
import torchvision.transforms.functional as F

import numpy as np
import time

import json
import numpy as np
import skimage.io
import math
import scipy

from PIL import Image
import cv2

import pickle
import argparse
import warnings

#%%
from geojson import MultiPolygon, Feature, FeatureCollection, dump, Point
from osgeo import gdal, gdalnumeric, ogr
from geojson.geometry import Point

#%% SAM
from segment_anything.bbox_prompt_mask_generator import SamBBoxMaskGenerator
from segment_anything import sam_model_registry

#%%

current = os.getcwd()
titaniach = Path(current.split('Caroline')[0]) / 'Caroline'



#%% define function

# define segmentation and mask plotting routine

def segmask_box(img, target, score_thresh=0.5, alpha=0.8, colors=None, width=1, fontsize=7, font=None, del_pxs=10):
    """
        img: torch.tensor or numpy.array, should have shape (H, W, C)
        target: dictionary; output of dataloader
                target['masks'] and target['boxes'] must be defined.
                NEW in this file: target['masks'] can contain sparse torch tensors
        alpha: alpha value for segmentations masks
        colors: specify color (but this is not properly implemented yet)
        width: width to draw bounding box

        returns an image with segmentation masks and bounding boxes drawn on it

        for testing: 
        target = predictions[idx]
        img = tiles[idx]

    """

    # Handle Grayscale images
    if img.shape[0] == 1:
        img = torch.tile(img, (3, 1, 1))
    # check if shape is correct (H, W, C)
    if not img.shape[0] == 3 or img.shape[0] == 1:
        raise Warning("please help me with the shape. Current shape is: {}, but I expect (H, W, C)".format(img.shape))

    # the image can be a torch tensor with range 0-1 and type float
    # it gets converted to 0-255 uint8
    img = torch.tensor(np.asarray(F.to_pil_image(img))) # (C, H, W)

    if 'scores' in target.keys():
        # get scores (they are ordered!) and cut off at threshold
        scores = target["scores"]
        cutoff_idc = torch.where(scores > score_thresh)
        # example (tensor([0, 1, 2, 3], device='cuda:0'),)
        # get masks and boxes
        masks = torch.index_select(target['masks'], 0, cutoff_idc[0]) # target["masks"][cutoff_idc]
        boxes = target["boxes"][cutoff_idc]
        labels = target["labels"][cutoff_idc]
        scores = target["scores"][cutoff_idc]

    else:
        # get masks and boxes
        masks = target["masks"]
        boxes = target["boxes"]
        labels = target["labels"]

    out_dtype = torch.uint8
    # define colors
    num_boxes =  masks.shape[0]
    if colors is None:
        # colors = torchvision.utils._generate_color_palette(num_boxes)
        # actually, get colors depending on the label
        # bands, 1 --> green, #008000 (matplotlib.colors.to_hex('green'))
        # double ridges, 2 --> maroon, #800000
        # ridge complexes, 3 --> deepskyblue, #00bfff
        # undifferentiated linea, 4 --> khaki, #f0e68c
        color_dict = {0: 'black', 1: '#ED9A22', 2: 'maroon', 3: 'deepskyblue', 4: 'khaki' } # in hex: 
        # to list
        colors = [color_dict[label.item()] for label in labels]

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        colors_.append(torch.tensor(color, dtype=out_dtype))

    # convert bounding boxes to list
    img_boxes = boxes.to(torch.int64).tolist()

    img_to_draw = img.detach().clone() # clone img to draw on it

    # del_idx = []

    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        # index correctly
        img_to_draw[np.array(mask.coalesce().indices()[0]), np.array(mask.coalesce().indices()[1])] = color[None, :]

    out = img * (1 - alpha) + img_to_draw * alpha
    out = out.to(out_dtype)

    # save this mask image alone
    out_masks = out.clone()

    # bounding boxes
    img_for_bbox = F.to_pil_image(out.permute(2,0,1))
    draw = ImageDraw.Draw(img_for_bbox) #  Image.fromarray(out.cpu().numpy())

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=fontsize)
    # txt_font = ImageFont.truetype("LiberationSansNarrow-Regular.ttf", size=fontsize)
    draw.fontmode = "L"
    #ImageFont.load_default()

    # here, we need colors in int or tuple format
    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            # get category from class dict
            margin = width + 1
            try:
                draw.text((bbox[0] + margin, bbox[1] + margin), str(label.item()), fill=(0,0,0), font=txt_font) # text in black
                # or print cat_dict[label.item()] instead of label
            # note: to make text white, just get rid of fill=color
            except KeyError:
                # then we have an undefined category
                undef_label='undefined'
                draw.text((bbox[0] + margin, bbox[1] + margin), undef_label, fill=(0,0,0), font=txt_font)
    result = torch.from_numpy(np.array(img_for_bbox)).permute(2, 0, 1).to(dtype=torch.uint8)

    return result, out_masks

def get_model_instance_segmentation(num_classes, pretrained=True, posweights=None, minsize=100, maxsize=500):
    '''
        generates a model that can be filled with weights
    '''
    # print('posweights are {}'.format(posweights))
    # send to device
    if not isinstance(posweights, type(None)):
        # if there are weights, send them to the device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        posweights.to(device)
    print('minsize: {}, maxsize: {}'.format(minsize, maxsize))
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained,
                        image_mean=[0.485, 0.456, 0.406],
                        image_std=[0.229, 0.224, 0.225],
                        min_size=minsize, # args.minsize, # 100
                        max_size=maxsize, # 500
                        posweights=posweights # implemented in site_packages, python lib, envs
                        )
    # note: you can pass all **kwargs that can be passed to MaskRCNN as well!
    # from C:\Users\ch20s351\Anaconda3\envs\pytorch\Lib\site-packages\torchvision\models\detection\mask_rcnn.py
    # model = MaskRCNN(backbone, num_classes, **kwargs)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_sam_model(num_classes, img_size, ckpt_path):
    '''
    SAM, LineaMapper v2.0
    - ckpt_path (checkpoint path, e.g. './ckpts/instseg_bb.pt')

    Get result with:
    # generating the masks from bounding box inputs
    sam_result = mask_generator.generate(image, boxes)

    # The result is a dict with the following keys:
    # - 'segmentation': np.ndarray with the mask
    # - 'bbox': np.ndarray with the bounding box in xywh format
    # - 'predicted_iou': float with the predicted iou

    instances = (sam_result['segmentation'][:, 1:].sum(axis=1) > 0) # of shape (N, 224, 224) with N number of masks

    '''
    sam = sam_model_registry['vit_b'](checkpoint=ckpt_path, num_classes=num_classes, image_size=img_size)
    mask_generator = SamBBoxMaskGenerator(sam)

    return mask_generator

def get_rcnnSAM(rcnnsam_args):
    '''    
    e.g.
    rcnnsam_args = {
        'num_classes': 5,
        'img_size': 224,
        'ckpt_path': './ckpts/instseg_bb.pt',
        'model_name': "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt",
        'minsize': 200,
        'maxsize': 300, 
        'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    }
    '''
    mask_generator = get_sam_model(rcnnsam_args['num_classes'], rcnnsam_args['img_size'], rcnnsam_args['ckpt_path'])
    box_generator = load_LineaMapper(rcnnsam_args['model_name'], rcnnsam_args['num_classes'], rcnnsam_args['minsize'], rcnnsam_args['maxsize'], rcnnsam_args['device'])

    return mask_generator, box_generator

def get_rcnnsam_output(mask_generator, box_generator, image_formaskrcnn):
    '''
    image: torch tensor (float), shape (C, H, W) (for mask r-cnn)
    TODO: batch processing
    '''
    # get mask r-cnn result
    result = box_generator([image_formaskrcnn])
    boxes = result[0]['boxes']

    # prompt sam with mask r-cnn boxes (faster r-cnn)
    image_forsam = (255*image_formaskrcnn).permute(1,2,0).numpy().astype('uint8') # for SAM, input is (H, W, C)
    sam_result = mask_generator.generate(image_forsam, boxes)

    # update result with sam masks
    # this only works if result is not empty
    masks = torch.as_tensor(sam_result['segmentation'][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension
    result[0]['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
    # return the updated result
    return result


def moving_window_tiling_ref(arr, row_step_n, col_step_n, tile_size, sort_zero=False, pad=False, xoffset=0, yoffset=0):
    '''
        pad: If True, the output tiles in the last row and column, where usually not a full tile shift fits in, is padded with zeros (in order to stack them).
             If False, the output tile size is conserved and more overlap is generated for the last row and column.
    example for tests:
    row_step_n = 100
    col_step_n = 100
    tile_size = 110
    sort_zero=False
    pad=True
    '''
    img_length = arr.shape[1]
    img_height = arr.shape[0]
    # calculate row and col partitioning
    # img_height - 1 not to run into 'out of bounds' index error --> but I think this leads later to other problems!
    rowsp = (img_height   - tile_size)/row_step_n # e.g. = 2.97, then we shift two full times plus one time only 0.97 of row_step
    colsp = (img_length  - tile_size)/col_step_n

    # initialize list of arrays
    tiles = []
    positions = [] # to store the x- and y-position of the moved file
    # end-of-row indicator
    end_of_row = False
    end_of_col = False

    for row_idx_n in range(math.ceil(rowsp)+1):
        # print(row_idx_n)
        # round up and test in every loop if row_idx is more than 1 away of 'rowsp' value
        if row_idx_n < rowsp:
            # print('row index: {}'.format(row_idx))
            row_idx = row_idx_n
        else:
            # print('end of row')
            end_of_row = True # set indicator to true
            if pad:
                row_idx = row_idx_n
            else:
                row_idx = rowsp # int((rowsp - row_idx_n + 1)*100)
                # print('row_idx for last row: {}'.format(row_idx))

        for col_idx_n in range(math.ceil(colsp)+1): # goes from 1,2,3 if colsp is 2.79 for example
            # print(col_idx_n)
            if col_idx_n < colsp:
                # print(col_idx_n)
                col_idx = col_idx_n
            else:
                # print('end of column')
                end_of_col = True # set indicator to true
                if pad:
                    col_idx = col_idx_n
                else:
                    # print('col_index: {}'.format(col_idx))
                    col_idx = colsp # int((colsp - col_idx + 1)*100)
                    # print('col_idx for last column: {}'.format(col_idx))

            positions.append((round(row_idx*row_step_n) + xoffset, round(col_idx*col_step_n) + yoffset))
            # print(positions)

            if (end_of_col or end_of_row) and pad:
                # for every end of row and if we would like to pad, treat specially with np.pad
                # print('padding')
                # I am sure there is a slightly better way of doing this, for example by using a variable for the 'tile_size'  
                if end_of_row and not end_of_col:
                    temp_tile = arr[round(row_idx*row_step_n):, round(col_idx*col_step_n):round(col_idx*col_step_n + tile_size)]
                    pad_rows = tile_size - temp_tile.shape[0] # the number of rows
                    pad_cols = 0
                elif end_of_col and not end_of_row:
                    temp_tile = arr[round(row_idx*row_step_n):round(row_idx*row_step_n + tile_size), round(col_idx*col_step_n):]
                    pad_rows = 0
                    pad_cols = tile_size - temp_tile.shape[1] # the number of cols
                elif end_of_row and end_of_col:
                    temp_tile = arr[round(row_idx*row_step_n):, round(col_idx*col_step_n):]
                    pad_rows = tile_size - temp_tile.shape[0] # the number of rows
                    pad_cols = tile_size - temp_tile.shape[1] # the number of cols
                # pad now
                # print(pad_rows, pad_cols)
                tile = np.pad(temp_tile, ((0, pad_rows), (0, pad_cols)), 'constant', constant_values=(0, 0) ) # Number of values padded to the edges of each axis. ((before_1, after_1), ... (before_N, after_N))
            else: # normal indexing
                tile = arr[round(row_idx*row_step_n):round(row_idx*row_step_n + tile_size),
                        round(col_idx*col_step_n):round(col_idx*col_step_n + tile_size) ]
            

            if sort_zero == True:
                # check if tile is totally empty. We do not want zero arrays! just do not append them and give a warning
                if np.equal(tile, np.zeros((tile_size, tile_size))).all() == False: # then, array is not equal to np.zeros() and we do append
                    tiles.append(tile)
            else:
                # just append
                # print(tile.shape)
                tiles.append(tile)

            # set col indicator to False again
            end_of_col = False
        # set row indicator to False again
        end_of_row = False

    # return an array with the tiles in the first axis
    if len(tiles) == 0:
        raise Warning
    else:
        return np.stack(tiles), positions
    
def mask_iou(mask1, mask2, mask_thresh=0.5):
    # intersection is where both masks are higher than threshold
    intersection = (mask1>mask_thresh) & (mask2>mask_thresh)
    # union is where either mask1 or mask2 are higher than threshold
    union = (mask1+mask2)>mask_thresh
    # iou is calculated by dividing the intersecting points by the unified points
    if np.count_nonzero(intersection)>0:
        iou = np.count_nonzero(intersection)/np.count_nonzero(union)
    else:
        # define iou=0 if no intersection at all
        iou=0
    return iou

# function from Z:\Groups\PIG\Caroline\lineament_detection\galileo_manual_segmentation\code\science_with_manual_segmentations\science_with_manual_segs_algo.py
def mesh(shape0, shape1):
    # generates meshgrid
    xline = np.arange(0, shape0)
    yline = np.arange(0, shape1)
    x, y = np.meshgrid(xline,yline)
    return x,y

# from Z:\Groups\PIG\Caroline\isis\Europa_code\python_code\Gimp_layers_to_geojson.py 
# define functions for conversion of indices in array to geo-coordinates for QGIS polygon
def xidx_to_xcoord(xpx, xres, ulx):
    return (xpx * xres) + ulx

def yidx_to_ycoord(ypx, yres, uly):
    return (ypx * yres) + uly


def load_LineaMapper(model_name, num_classes, minsize, maxsize, device):

    # load model with specified path
    # savep = Path('model_dict')
    # get the model
    model = get_model_instance_segmentation(num_classes, minsize=minsize, maxsize=maxsize)
    # move model to the right device
    model.to(device)
    # model_path = savep.joinpath(model_name)
    model.load_state_dict(torch.load(model_name, map_location=device))
    model.eval()

    return model


def geotiff_to_arr(parsedargs):
    '''
        Reads a full path to a TIFF file.
        Opens it with GDAL as a dataset.
        Normalizes the array
        Sets fill values to -0.01
    '''
    geotiff_path = Path(parsedargs.geofile)
    print(geotiff_path)
    # open geotiff
    dataset = gdal.Open(geotiff_path.as_posix(), gdal.GA_ReadOnly)  #, gdal.GA_ReadOnly not to change the file!

    # transform to numpy array
    arr = dataset.ReadAsArray()
    arr = np.array(arr, dtype='float32')
    # there are fill values (=-3.4028227e+38) we want to get rid of
    # now it can happen that 0 is 'the wrong minimum', and the contrast gets small
    # therefore, we can't simply use arr[arr<0] = 0
    # but we want to figure out the minimum and the maximum and rescale that to 0-1, in the best case.
    arrmin = np.min(arr[arr>0])
    # first, get rid of fill values
    arr[arr<0] = arrmin - 0.01 # the minus 0.01 makes sure that we are smaller than the minimum
    # lets' rescale now
    arr = (arr - arr.min())/(arr-arr.min()).max()

    if parsedargs.test:
        # for testing: take an excerpt that you can handle well
        arr = arr[:500, :500]
        # arr = arr[300:, :800] # for older tests with E6ESCRATER01_GalileoSSI_Equi-cog.tif

    if parsedargs.plot:
        plt.imshow(arr)
        plt.show()

    if len(arr.shape) > 2:
        print('WARNING: I only take the first band. Fingers crossed it is the right one. Otherwise please implement functionality of RGB channels.')
        arr = arr[0]
        
    return arr, dataset

def sparse_mask_iou(mask1, mask2):
    '''
        input: two sparse torch tensors, must be coalesced.
        output: intersection-over-untion of these two masks

    can be tested with:
    testshape = [5,8]
    # define mask1
    idcs1x =  torch.tensor([[0,0,1,1,1,1,2,2,2,3,3,3],[4,5,2,3,4,5,1,2,3,0,1,2]])
    v1 = torch.ones(idcs1x.shape[1], dtype=torch.float32)
    mask1 = torch.sparse_coo_tensor(idcs1x, v1, testshape).coalesce()
    # define mask2
    idcs2x =  torch.tensor([[0,0,1,1,2,2,3,3],[2,3,3,4,4,5,5,6]])
    v2 = torch.ones(idcs2x.shape[1], dtype=torch.float32)    
    mask2 = torch.sparse_coo_tensor(idcs2x, v2, testshape).coalesce()
    '''
    # take indices from sparse masks and convert to lists with entries like (1, 3), (1, 4),  (1, 5), that define non-empty pixels!
    tuples1 = list(zip(mask1.indices()[0].tolist(), mask1.indices()[1].tolist() ))
    tuples2 = list(zip(mask2.indices()[0].tolist(), mask2.indices()[1].tolist() ))
    # find intersection of these two lists:
    intersets = set(tuples1).intersection(tuples2)
    # union is total amount of cells of two masks, minus one time the intersection (otherwise we count this part twice)
    union = len(tuples1) + len(tuples2) - len(intersets)
    # iou is calculated by dividing the intersecting points by the unified points
    if len(intersets)>0:
        iou = len(intersets)/union
    else:
        # define iou=0 if no intersection at all
        iou=0
    return iou, intersets, union, len(tuples1), len(tuples2) # 2024-06-03: also return intersections and unions, and length of masks, for another merge crit.

def batch_process_rcnnsam(mask_generator, box_generator, images_formaskrcnn):
    '''
    images: LIST OF torch tensor (float), shape (C, H, W) (for mask r-cnn)
    batch processing!
    '''
    # get mask r-cnn result
    result = box_generator(images_formaskrcnn)
    # # extract list of boxes:
    # boxes = []
    # for res in result:
    #     boxes.append(res['boxes'])
    # torch.stack(boxes)

    # prompt sam with mask r-cnn boxes (faster r-cnn)
    for res_idx, im in enumerate(images_formaskrcnn):
        if len(result[res_idx]['boxes']) == 0: # empty boxes..
            # print('empty box')
            continue
        # images_forsam = [(255*im).permute(1,2,0) for im in images_formaskrcnn] # for SAM, input is (H, W, C)
        # sam_result = mask_generator.generate(torch.stack(images_forsam), torch.stack(boxes))
        image_forsam = (255*im).permute(1,2,0).numpy().astype('uint8') # for SAM, input is (H, W, C)
        sam_result = mask_generator.generate(image_forsam, result[res_idx]['boxes'])

        # update result with sam masks
        masks = torch.as_tensor(sam_result['segmentation'][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension
        result[res_idx]['masks'] = torch.as_tensor(masks, dtype=torch.float32)
        # masks.shape: torch.Size([100, 1, 224, 224])
        # masks.dtype: torch.float32        
        
    # return the updated result
    return result


def get_predictions(obj):
    '''
        obj is the 'self' passed from the class LineaMapper. It contains num_tiles, batch_size, device, model, mask_threshold, del_pxs, test, tiles
        
        batches the tiles and returns predictions
        here, the masks get thresholded with mask_threshold, empty masks get filtered out

        2024-06-01: implementation of SAM for the masks

        Future: implement num_predictions (default is now max = 100 predictions, which might me a bit too much.)
    '''
    predictions = []

    for bidx in range(math.ceil(obj.num_tiles/obj.batch_size)):
        print(bidx)
        # make batch (list of images)
        # if we are at the end, there is no index error! But all is fine, I checked.
        batch = [x.to(obj.device) for x in obj.tiles[bidx*obj.batch_size:(bidx+1)*obj.batch_size]]

        # image must be a torch tensor, not numpy
        with torch.no_grad():
            # see below # prediction = obj.model(batch)
            # prompt SAM with Mask R-CNN boxes
            prediction = batch_process_rcnnsam(obj.mask_generator, obj.box_generator, batch)
        # move back to cpu
        for pred in prediction:

            # threshold mask and convert to bool, then make sparse mask out of it 
            # because operations on sparse masks are limited!
            pred['masks'] = pred['masks'][:,0,:,:] # from shape torch.Size([100, 1, 224, 224]) to torch.Size([100, 224, 224])
            pred['masks'][pred['masks']<obj.psags.mask_threshold] = 0 # mask is currently between 0 and 1
            # now make boolean out of it
            pred['masks'] = 1*pred['masks'].bool()
            # test for empty masks
            # use count_nonzero on width and height dim, so we end up with a tensor with the length of the number of dimensions
            # e.g. tensor([ True,  True, False,  True, False, False,  True, False,  True,  True, ...])
            keep_idcs = torch.count_nonzero(pred['masks'], dim=(1,2)) > obj.psags.del_pxs
            # apply mask to keep_indices
            pred['masks'] = pred['masks'][keep_idcs]
            pred['masks'] = pred['masks'].to('cpu').to_sparse().coalesce()
            pred['labels'] = pred['labels'][keep_idcs].to('cpu')
            pred['scores'] = pred['scores'][keep_idcs].to('cpu')
            pred['boxes'] = pred['boxes'][keep_idcs].to('cpu')
        # we append to 'predictions' list. 
        # prediction = prediction.to('cpu') # move back to cpu? ever needed?
        predictions += [x for x in prediction]

        torch.cuda.empty_cache() # to free up memory

    # visualisation
    if obj.psags.test:
        # index predictions and tiles
        idx = 0
        #for idx in range(len(predictions)):
        t, masks = segmask_box(obj.tiles[idx], predictions[idx], score_thresh=0.5, alpha=0.8, colors=None, width=1, fontsize=6, font=None, del_pxs=obj.psags.del_pxs)

        fig, ax = plt.subplots(ncols=2, figsize=(20,40)) # ncols=4
        # raw image
        ax[0].imshow(obj.tiles[idx].permute(1,2,0))
        # mask
        # ax[1].imshow(masks)
        # masks and bounding boxes
        ax[1].imshow(t.moveaxis(0,2))

    return predictions

def reference_predictions_to_img(obj):
    '''
    obj is the 'self' passed from the class LineaMapper. It contains predictions, class_scores_dict and geofile and test and plot (in .psags), positions, arr

    save predictions with respect to (0,0) coordinate of untiled image
    basically, I have to shift the box coordinates
    boxes have original coords: xmin, ymin, xmax, ymax
    but with coords relative to indices: ymin, xmin, ymax, xmax (I don't fully understand where it is swapped)

    at the same time:
    put all predictions into one full prediction (meaning that in the end, we only have one dictionary!)
    this happens inside the for loop

    And: only predictions higher than the class score threshold are returned. Others get deleted here

    TODO: use broadcasting in this code!

    # for debugging, you can run the main function with pargs = parser.parse_args([...])
    predMapper = LineaMapper(pargs)
    predMapper.class_scores_dict = {1: 0.3, 2: 0.4, 3: 0.6, 4: 0.7}
    predMapper.forward()
    obj = predMapper
    '''
    # initialize lists
    full_preds_masks_list = []
    full_preds_boxes_list = []
    full_preds_scores_list = []
    full_preds_labels_list = []

    full_pred = {}

    for pidx, pred in enumerate(obj.predictions):
        # debugging
        # print('before: {}'.format(pred["scores"]))
        # if pidx%100 == 0:
            # print(pidx)
            # print(time.time() - start_time_preds_loop)
        # filter for score threshold first to reduce amount of predicted instances
        # index with score_threshold. Keep only predictions above score threshold
        # implement different score thresholds for different classes! with class_scores_dict
        cutoff_idc_list = []
        # go through all labels (e.g. 1,2,3,4)
        for labl in pred['labels'].unique().tolist():
            # # debugging:
            # print('the label is: {}'.format(labl))
            # print('class score for this label is: {}'.format(obj.class_scores_dict[labl]))
            # print('this is the array for scores higher than the object class score: {}'.format(pred["scores"] >= obj.class_scores_dict[labl]))
            # print(pred["scores"] )
            # print(pred['labels'])

            # we want the indices of entries that fulfill that the score is higher than the pre-defined class threshold AND that the label is only the selected label
            # merge the two lists:
            cutoff_idc_list = cutoff_idc_list +  torch.where( (pred["scores"] >= obj.class_scores_dict[labl]) & (pred['labels'] == labl) )[0].tolist()
            # print(cutoff_idc_list)
        # cast to tensor again
        cutoff_idc_list = torch.tensor(cutoff_idc_list)

        # test if we need to get add anything, i.e. if cutoff_idc is not empty
        # else, we do not need any prediction from this loop
        if len(cutoff_idc_list) > 0:
            # print((pred['masks']).shape)
            # example (tensor([0, 1, 2, 3], device='cuda:0'),)
            pred['masks'] = torch.index_select(pred['masks'], 0, cutoff_idc_list).coalesce()
            # print((pred['masks']).shape)
            # get masks and boxes
            # masks = full_pred["masks"][cutoff_idc_list]
            # to index the sparse mask array, we just need to throw out the indices in the first dimension that are not in cutoff_idc
            pred["boxes"] = pred["boxes"][cutoff_idc_list]
            pred["labels"] = pred["labels"][cutoff_idc_list]
            pred["scores"] = pred["scores"][cutoff_idc_list]
            # print('after: {}'.format(pred["scores"]))

            # # check scores
            # for score in pred["scores"]:
            #     if score < 0.3:
            #         print('SCORE BELOW threshold')
            #         print(score)
            #         print(pred['scores'])
            #         print(cutoff_idc_list)
            # print('pred scores before shifting: {}'.format(pred['scores']))
            # print(cutoff_idc_list)
            
            # pred['boxes'] have shape [num_boxes, 4], e.g. torch.Size([100, 4])
            xoff, yoff = obj.positions[pidx]
            # print(xoff, yoff)
            # make a tensor of broadcastable form to shift bounding box coords
            offset = torch.tensor([yoff, xoff, yoff, xoff]).expand(len(pred['boxes']), -1)
            # offset has now shape: torch.Size([100, 4])
            pred['boxes'] = pred['boxes'] + offset

            # we construct a new sparse tensor
            # since the padding is done with zeros, we can use 'sparse' to store the tensor to optimise memory consumption
            # the indices tensor has e.g. shape torch.Size([3, 37290])
            # in pred['masks'], the number of masks (pred['masks'].indices()[0]), the x- (pred['masks'].indices()[1]) and the y-direction (pred['masks'].indices()[2]) are stored.
            # therefore, we want to shift [1,2] by [xoff, yoff]            
            # correcting positions: (not sure why. I seem to not understand something small)
            if xoff == 0:
                xoffcorr = xoff
            else:
                xoffcorr = xoff -1
            if yoff == 0:
                yoffcorr = yoff
            else:
                yoffcorr = yoff -1
            shifted_x = pred['masks'].coalesce().indices()[1] + xoffcorr
            shifted_y = pred['masks'].coalesce().indices()[2] + yoffcorr
            # print(xoff)
            # for hg in range(len(pred['masks'].coalesce().to_dense())):
            #     print((pred['masks'][hg].coalesce().indices()[1] + xoff -1).min())
            #     plt.imshow(pred['masks'].coalesce().to_dense()[hg].cpu())
            #     plt.show()
            # print(shifted_x.min())
            # print(shifted_y.min())

            shifted_indices = torch.cat( (pred['masks'].coalesce().indices()[0][None, :], shifted_x[None, :], shifted_y[None, :]), dim=0 )
            # the values stay the same
            # new shape: (number of masks, full_array shape x, full_array shape y,)
            new_shape = (pred['masks'].shape[0], obj.arr.shape[0], obj.arr.shape[1])
            pred['masks'] = torch.sparse_coo_tensor(shifted_indices.tolist(), pred['masks'].values().tolist(), new_shape)
            
            # print('pred scores after shifting: {}'.format(pred['scores']))
            # append to lists
            full_preds_masks_list.append(pred['masks'])
            full_preds_boxes_list.append(pred['boxes'])
            full_preds_scores_list.append(pred['scores'])
            full_preds_labels_list.append(pred['labels'])

    print('concatenating now')
    # concatenate
    if len(full_preds_boxes_list) > 0: # otherwise, torch.cat fails
        full_pred['boxes'] = torch.cat( full_preds_boxes_list, dim=0)
        full_pred['labels'] = torch.cat( full_preds_labels_list, dim=0)
        full_pred['scores'] = torch.cat( full_preds_scores_list, dim=0)
        full_pred['masks'] = torch.cat( full_preds_masks_list, dim=0).coalesce()

        if obj.psags.test:
            # show on full image:
            # we need an rgb image
            img = np.repeat(obj.appimg[np.newaxis, :, :], 3, axis=0)
            t, masks = segmask_box(torch.tensor(img), full_pred, score_thresh=0, alpha=0.8, colors=None, width=1, fontsize=6, font=None)

            # save a small image for preview
            fig, ax = plt.subplots(ncols=2, figsize=(80,40)) # ncols=4
            # raw image
            ax[0].imshow(torch.tensor(img).permute(1,2,0))
            # mask
            # ax[1].imshow(masks)
            # masks and bounding boxes
            ax[1].imshow(t.moveaxis(0,2))

            fig.savefig((titaniach / 'lineament_detection/pytorch_maskrcnn/output/full_imgs').joinpath(obj.geotiff_filename + '_preview.pdf'), bbox_inches='tight')
            if obj.psags.plot:
                fig.show()

    return full_pred

# 2024-01-26, from LineaMapper_to_img_sparsemasks_experimental.py,
# tested well!
def fit_line_to_mask(masktf):
    '''
    input is a mask to fit (masktf), a torch sparse tensor
    '''
    # first, generate a 'contours'. list of numpy arrays
    ftl_idcs = [ np.array([[y, x]]) for x, y in  zip(masktf.coalesce().indices()[0].tolist(), masktf.coalesce().indices()[1].tolist() ) ]
    # stack to array of shape [Npoints, 1, 2] (2 for the x/y pair)
    ftl_idcs = np.stack(ftl_idcs)
    # fit the line with the off-the-shelve openCV function fitLine
    [vx,vy,x,y] = cv2.fitLine(ftl_idcs, cv2.DIST_L2,0,0.01,0.01)
    # calculate the angle
    alpha = np.arctan(vy/vx) # alpha is in radians
    return 90 + alpha*180/np.pi # change radians to degrees. Add 90 degrees to orient it with north = 0°

def azimuth_merge_crit(mask1, mask2):
    '''
        This function checks if the azimuth difference of the two 'to be merged' masks is within a given range (obj.azimuth_range)
        It outputs the azimuth difference
    '''

    # calculate azimuth of first mask
    azimuth1 = fit_line_to_mask(mask1)
    # and of second mask
    azimuth2 = fit_line_to_mask(mask2)

    # retun azimuth difference
    return abs(azimuth1 - azimuth2)

def calculate_mask_iou_for_merge_crit(obj):
    '''
        calculates the intersection over the union times the multiplication factor (to account for non-overlapping areas)
        for masks indices that showed a positive box IoU

        Also calculates the difference in the direction (azimuth) of two lineaments.
        and automatically merges if one mask is almost entirely covered by the other.

        obj is the 'self' passed from the class LineaMapper. It contains multiplication_factor and iou_thresh inside .psags, and masks and boxes
    '''

    # get overlapping boxes:
    boxiou_mat = box_iou(obj.boxes, obj.boxes)
    # get nonzero elements on diagonal
    box_iou_indcs = (boxiou_mat>0.1).nonzero().tolist()

    merge_idc = []
    # only loop through the indices that show a positive box_iou
    for i,j in box_iou_indcs: 
        if i == j:
            # skip diagonal entries (is there a better way?)
            continue

        # calculate mask IoU and multiply it by a factor that was chosen by the user
        masks_iou, intersets, union, lenA, lenB = sparse_mask_iou(obj.masks[i].coalesce(), obj.masks[j].coalesce())
        miou = obj.psags.multiplication_factor*masks_iou
        
        # furthermore, check if the two azimuths are the 'the same', within errorbars of obj.psags.azimuth_diff_range
        # calculate difference
        azimuth_difference = azimuth_merge_crit(obj.masks[i], obj.masks[j])

        # check if one of the masks is contained (almost entirely) in the other:
        # this is the case if the number of pixels of the mask is almost or entirely the same as the intersection
        # merge if mask A is contained in B or mask B is contained in A with 10% overlap accuracy
        if ((lenA <= intersection * 1.1) & (lenA >= intersection * 0.9)) or ((lenB <= intersection * 1.1) & (lenB >= intersection * 0.9)): # and (azimuth_difference <= obj.psags.azimuth_diff_range):
            merge_idc.append((i,j))

        # add masks to merge_idc list if mask_iou is bigger than iou_thresh for debugging
        # if the differnce between azimuth1 and azimuth2 is withing the azimuth_range,
        # then add the tuple.
        elif (miou >= obj.psags.iou_threshold) and (azimuth_difference <= obj.psags.azimuth_diff_range):
        # if (azimuth_difference <= obj.psags.azimuth_diff_range):
            merge_idc.append((i,j))

    return merge_idc

def get_clean_merge_lists(obj):
    '''
        From all masks that passed the threshold criterion, this function generates clean (only unique values) lists with indices to merge
        e.g. [1], [2,4,28], [3], [...]
    
        obj is the 'self' passed from the class LineaMapper. It contains multiplication_factor masks and merge_idc
    
    '''
    # initialize list with lists of indices to merge
    final_lists = []

    temp_list = []
    merg_list = []

    for p in range(len(obj.masks)):
        # let's first see if there is a list where our index is appearing 
        # this is a special case,
        for flist in final_lists:
            if p in flist:
                merg_list = flist

        if merg_list == []: # then, merge list is empty and we start 'fresh'
            # we for sure append the index itself, to keep non-overlapping masks
            temp_list.append(p)
            # next, we look for direct overlaps
            for i,j in obj.merge_idc:
                if i == p:
                    # for the current index, we append the tupled index to the temporary list
                    temp_list.append(j)

            # after this iteration, we add temp_list to final_lists and re-initialize temp_list 
            final_lists.append(temp_list)
            temp_list = []
        else: # then, there is already a list to which we can append
            # here, we use the fact that lists are mutable
            for i,j in obj.merge_idc:
                if i == p:
                    if j not in merg_list: # to reduce duplicate entries
                        # for the current index, we append the tupled index to the temporary list
                        merg_list.append(j)  
            merg_list=[]

    return final_lists



def merge_bool_masks(obj):
    '''
        obj is the 'self' passed from the class LineaMapper. It contains final_lists, labels, scores, masks

        Merges boolean masks according to final_lists
        
        it returns a dictionary with merged predictions for the full image
    '''

    # merge boolean masks

    merged_arrs = []
    merged_labels = []
    merged_scores = [] # averaged scores (above score_threshold, since we filtered with it before)
    merged_boxes = []

    for mlist in obj.final_lists:
        # initialize merged_mask with first index of mlist
        merged_mask = torch.index_select(obj.masks, 0, torch.tensor([mlist[0]]) ).coalesce()
        # if mlist has more than one entry, we add to the sparse mask simply with +=
        if len(mlist) > 1:
            for mlindx in mlist:
                merged_mask += torch.index_select(obj.masks, 0, torch.tensor([mlindx], dtype=torch.int64) ).coalesce() # like masks[mlist], but for sparse tensors
            
        # append boolean array of summed boolean arrays to merge
        # attention: we take index 0
        # merged_mask = (torch.sum(arrs_to_merge, dim=0)[0]).bool()
        # if torch.count_nonzero(merged_mask) > 0:
        merged_arrs.append(merged_mask)
        # array to box:
        pos = np.array(merged_mask.indices()[1]), np.array(merged_mask.indices()[2])
        merged_boxes.append([np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])])
        # labels: most common 
        values, counts = np.unique(obj.labels[mlist], return_counts=True)
        ind = np.argmax(counts)
        merged_labels.append(values[ind])
        # scores: average
        merged_scores.append(np.average(obj.scores[mlist]))

    # stack merged mask arrays
    memars = torch.stack(merged_arrs).coalesce() # shape torch.Size([140, 450, 1158])
    # squeeze middle dimension by defining new sparse tensor (don't know how else to do this)
    # idcs1x, v1, testshape
    memars = torch.sparse.sum(memars, dim=1)

    # to dict for displaying
    merged_pred = {
                    "boxes": torch.as_tensor(np.array(merged_boxes), dtype=torch.float32), # create boxes from masks... np.vstack(ttarr[:,2]
                    "labels": torch.tensor(np.array(merged_labels, dtype=np.int16)),
                    "scores": torch.tensor(np.array(merged_scores, dtype=np.float32)),
                    "masks": memars
                    }

    return merged_pred

def merged_pred_to_pickle(obj):
    '''
        save dict 'merged_pred' to pickle file
    '''
    save_pkl_path = Path(obj.psags.savedir) / 'pickle_files'
    os.makedirs(save_pkl_path, exist_ok=True)

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_")
    with open(save_pkl_path.joinpath( dt_string  + obj.geotiff_filename + '_' + str(obj.sidx) + '.pickle'), 'wb') as handle:
        pickle.dump(obj.merged_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

def merged_pred_to_geojson(obj):
    '''
        save output to json file/pickle file

        obj is the 'self' passed from the class LineaMapper. It contains dataset, merged_pred, geotiff_filename, save_path, apppos (x- and y- offset)

    
    '''
    ulx, xres, _, uly, _, yres  = obj.dataset.GetGeoTransform() # not needed: xskew, yskew

    # # NEW: check if this part of the feature lies in the current 'field of view' of the subimg
    # # code outline
    # # define outside the field of view of the subimg in pixels
    # # it is not crucial that this computation is perfect! rounding errors are no problem
    # xoffset = obj.apppos[0]
    # yoffset = obj.apppos[1]
    # if (xoffset + yoffset) != 0:
    #     # NOTE: I had to exchange xoffset with yoffset. ulx is longitude, uly is latitude. swapped in python array
    #     newulx = xidx_to_xcoord(yoffset, xres, ulx)
    #     newlrx = xidx_to_xcoord(yoffset + obj.psags.cut_size, xres, ulx)
    #     newuly = yidx_to_ycoord(xoffset, yres, uly)
    #     newlry = yidx_to_ycoord(xoffset + obj.psags.cut_size, yres, uly)

    # initialize features
    features = []

    # loop through mask layers
    for midx in range(len(obj.merged_pred['masks'])):
        '''
            for testing:
            midx = 0
            layerimg = np.array(memars[0].to_dense())
        '''

        layerimg = np.array(obj.merged_pred['masks'][midx].to_dense())
        # calculate azimuth now:
        azimuth = fit_line_to_mask(obj.merged_pred['masks'][midx]) # we append later to feature! this is only a first proxy, as the projection might not be suited for azimuth extraction!
        # get polygon indices with cv2 function 'findContours'
        # this automatically detects contours
        contours, _ = cv2.findContours(layerimg.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # NOTE: this is the step that can take forever! simply due to the amount of masks!

        # append to list
        xarr = []
        yarr = []
        for idx in range(len(contours)):
            # TODO: delete small 'fuzzy' predictions
            # TEST well!

            parrx = contours[idx][:,0][:,0] # the shape of contours[0] is (npoints, 1, 2). I take here simply all points with the first [:,0] indexing, and the first or second entries with the second [:,0].
            parry = contours[idx][:,0][:,1]

            # for closing the polygon, append the first pair again
            # otherwise, I  need to fix geometries in QGIS, and can't import to ArcGIS
            parrx = np.append(parrx, contours[idx][0,0][0])
            parry = np.append(parry, contours[idx][0,0][1])

            # # parrx to xarr
            # if (xoffset + yoffset) != 0:
            #     xarr.append(xidx_to_xcoord(parrx, xres, newulx))
            #     yarr.append(yidx_to_ycoord(parry, yres, newuly))            
            # else:
            xarr.append(xidx_to_xcoord(parrx, xres, ulx))
            yarr.append(yidx_to_ycoord(parry, yres, uly))

        # to geojson Polygon feature
        # bring to format:
        # Polygon([[(2.38, 57.322), (23.194, -20.28), (-120.43, 19.15), (2.38, 57.322)]])
        # (this exact format is necessary)
        polyg = MultiPolygon([ [list(zip(xarre, yarre))] for xarre, yarre in zip(xarr, yarr)  ])

        # add feature (and id)
        # TODO: add correct id from layer name (first entry)
        id = int(obj.merged_pred['labels'][midx].item())
        score = round(obj.merged_pred['scores'][midx].item(), 2)  # we do want a float, not a string, rounded to two decimal places
        feature = Feature() # , properties={'id': id}
        feature['properties'] = {'id_int': id, 'score': score, 'comments': '', 'reviewed': 0, 'azimuth': azimuth}
        feature['geometry'] = polyg # NOTE: I always have to fix geometries... is there a way to overcome this? perhaps the polygon is not closed? i could solve this perhaps with this: https://gis.stackexchange.com/questions/263696/invalid-geometries-made-valid-dont-remain-valid 
        features.append(feature)

    # add the coordinate reference system (I copied this from an exported geojson file from QGIS ('Qgis_all_usgs_mosaics.qgz'))
    feature_collection = FeatureCollection(features, crs={'type': 'name',
    'properties': {'name': obj.dataset.GetProjection()}})

    # save as geojson
    # use date for saving to prevent writing errors
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_")
    # dt_ymd = now.strftime("%Y_%m_%d")
    save_json_path_dt = Path(obj.psags.savedir) / 'json_files'
    # make an individual folder for each day (to prevent a huge chaos!)
    os.makedirs(save_json_path_dt, exist_ok=True)
    with open(save_json_path_dt.joinpath(dt_string + obj.geotiff_filename + '_' + str(obj.sidx) + '.geojson'), 'w') as f:
        dump(feature_collection, f)

    return

def time_to_file(starttime, numprocesses, savepath, filename, sidx):
    '''
        This function simply writes the time in seconds used for this process to a file, together with the number of subprocesses (for LineaMapper, this is the number of tiles)
    '''
    print('For this image, I needed {:.2f} seconds, for {} tiles, so {:.2f} per tile.'.format(time.time() - starttime, numprocesses, (time.time() - starttime)/numprocesses))
    print(f'The time now is {datetime.now()}')

    # save time information to file
    os.makedirs(savepath, exist_ok=True)  
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_")
    with open(savepath.joinpath(dt_string  + filename + '_' + str(sidx) + "_processing_time.csv"), 'w') as f: 
        # header: time for process, number of processes, time per tile, filename
        f.write('time_for_process [s], number_of_processes, time_per_tile [s], filename\n')
        f.write('{:.2f}, {}, {:.2f}, {}'.format(time.time() - starttime, numprocesses, (time.time() - starttime)/numprocesses, filename))
    
    return


#%%

class LineaMapper():
    '''
        Note: The modelname can be a full path to the model weights or simply the filename if the file is in the same folder than this class (e.g. "Mask_R-CNN_pub2_run23_end_model.pt")
    '''
    def __init__(self, parsargs) -> None:
        # put individual score thresholds to dictionary
        assert len(parsargs.class_scores) == 4 # make sure there is exactly 4 entries
        '''
            parsargs: added azimuth_diff_range
        '''
        self.class_scores_dict = {k: v for k, v in zip(range(1,5), parsargs.class_scores)}
        
        # define geospecs (can be overwritten, of course)
        self.geosize = parsargs.geosize # default: 224
        self.row_step_n = int(self.geosize/2)
        self.col_step_n = int(self.geosize/2)
        self.batch_size = 9 # 9 is fine for 224 geosize
        self.psags = parsargs
        # use the GPU or the CPU, if no GPU is available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.minsize = parsargs.minsize # e.g. 200 # adjust for model
        self.maxsize = parsargs.maxsize # e.g. 300 # adjust for model
        self.modelname = parsargs.modelname # default e.g. "Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt"
        self.sampath = parsargs.sampath
        # self.cut_size = 800 # 3000 # TODO: calculate for each GPU/CPU setting individually, what 'too big' means. Calculate the cut_size.
        self.overlap = 70
        # define class_dict (hardcoded for LineaMapper v1.0.0)
        self.class_dict = {"band": 1, "double_ridge": 2, "ridge_complex": 3, "undifferentiated_linea": 4}
        self.num_classes = len(self.class_dict) + 1
        # rcnnSAM:
        self.rcnnsam_args = {
            'num_classes': self.num_classes,
            'img_size': self.geosize,
            'ckpt_path': self.sampath, # './ckpts/instseg_bb.pt',
            'model_name': self.modelname, # "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt",
            'minsize': self.minsize,
            'maxsize': self.maxsize, 
            'device': self.device,
        }

        # construct geotiff filename from full path
        # construct path and use stem
        self.geotiff_filename = Path(parsargs.geofile).stem
        print(self.geotiff_filename)

        # make savedirectory if it does not exist
        os.makedirs(self.psags.savedir, exist_ok=True)

        dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_")
        # save info file       
        self.info_path = Path(self.psags.savedir) / 'info'   
        os.makedirs(self.info_path, exist_ok=True)  
        with open(self.info_path.joinpath(dt_string  + self.geotiff_filename +".txt"), 'w') as f: 
            for key, value in self.__dict__.items(): 
                f.write('%s:%s\n' % (key, value))

    def get_tiles(self):
        # tile it up into 224x224 patches. in this step, I care for offsets due to a subimg tiling
        tiles, positions = moving_window_tiling_ref(self.appimg, self.row_step_n, self.col_step_n, self.geosize, xoffset=self.apppos[0], yoffset=self.apppos[1])
        num_tiles = len(tiles) # take the number of tiles now from the first 'shape' index (done with 'len') for the loop later
        # the array 'tiles' is of shape (number of tiles, geosize, geosize), e.g. (40, 224, 224)
        # let's convert to RGB
        tiles = np.repeat(tiles[:, np.newaxis, :, :,], 3, axis=1)
        # to tensor
        tiles = torch.tensor(tiles) # following the example, the shape is now: torch.Size([40, 3, 224, 224])
        return tiles, num_tiles, positions
    
    def display_preview(self, preds_to_display, displayname):
        # display a prediction with labels, bounding boxes (from masks)
        # show on full image:
        # we need an rgb image
        img = np.repeat(self.appimg[np.newaxis, :, :], 3, axis=0)
        t, _ = segmask_box(torch.tensor(img), preds_to_display, score_thresh=0, alpha=0.8, colors=None, width=1, fontsize=6, font=None, del_pxs=self.psags.del_pxs)

        fig, ax = plt.subplots(ncols=2, figsize=(40,20)) # ncols=4
        # raw image
        ax[0].imshow(torch.tensor(img).permute(1,2,0))
        # mask
        # ax[1].imshow(masks)
        # masks and bounding boxes
        ax[1].imshow(t.moveaxis(0,2))

        # use date for saving to prevent writing errors
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_")
        # dt_ymd = now.strftime("%Y_%m_%d")
        save_fig_path_dt = Path(self.psags.savedir) / 'preview'
        os.makedirs(save_fig_path_dt, exist_ok=True)
        fig.savefig(save_fig_path_dt.joinpath(dt_string + self.geotiff_filename + displayname), bbox_inches='tight')
        if self.psags.plot:
            fig.show()
        return


    def apply(self):
        '''
            appimg: the array or subarray. the image to which the following steps are applied:
            1. tiling
            2. get predicions
            3. reference predictions
            4. merge predictions
            5. save as geojson
        '''
        # tile up array and print info
        self.tiles, self.num_tiles, self.positions = self.get_tiles()
        print(f'The input image was divided into {self.num_tiles} tiles.')

        print(self)
        # get predictions 
        self.predictions = get_predictions(self)

        # reference predictions to (0,0)-coordinate of full image
        self.full_pred = reference_predictions_to_img(self)
        print('referenced predictions')
        # implement case without any predictions
        if len(self.full_pred.keys()) == 0:
            print('No predictions for this (sub)-image. Therefore, I continue.')
            return

        # I rename stuff (not really necessary)
        self.masks = self.full_pred["masks"]
        self.boxes = self.full_pred["boxes"]
        self.labels = self.full_pred["labels"]
        self.scores = self.full_pred["scores"] 

        # save this step (2023-06-20 edit)
        # self.display_preview(self.full_pred,  '_' + str(self.sidx) + '_full_prediction.pdf')

        # get indices that need to be merged
        self.merge_idc = calculate_mask_iou_for_merge_crit(self)
        print('merging...')

        # get lists that says which masks to merge
        self.final_lists = get_clean_merge_lists(self)
        # merge predictions with final_lists
        self.merged_pred = merge_bool_masks(self)
        print('merged!')

        # display predictions and save to img
        # self.display_preview(self.merged_pred, '_' + str(self.sidx) + '_merged_preview.pdf')

        # save merged_pred to pickle file
        merged_pred_to_pickle(self)

        # save to geojson (as Features)
        # this step also takes some time!
        merged_pred_to_geojson(self)

        # TODO: save as shapefiles

        # print time for this image in separate file (this is important for reporting results!)
        time_to_file(self.start_time_tiles, self.num_tiles, self.info_path, self.geotiff_filename, self.sidx)

        return


    def forward(self):
        self.start_time_full = time.time()
        print('class scores: {}'.format(self.class_scores_dict))
        self.arr, self.dataset = geotiff_to_arr(self.psags)

        total_num_tiles = [] # count tiles for csv info
        # load the model with the weights for LineaMapper (LineaMapper v1.1.0)
        # deprecated with rcnnSAM: self.model = load_LineaMapper(self.modelname, minsize=self.minsize, maxsize=self.maxsize, device=self.device)
        self.mask_generator, self.box_generator = get_rcnnSAM(self.rcnnsam_args)

        # new: split array into subimages (TODO: and stitch output together in the end)
        if np.max(self.arr.shape) > self.psags.cut_size:
            print('I have to cut the input image into sub-images.')
            # with the row_step, we can ensure a small overlap, which is important for connecting the features
            # I adapt the moving_window_tiling_ref in a way that allows for the last row and column to not repeat too much. I can simply cut them to ensure an overlap
            row_step_subi = self.psags.cut_size - self.overlap
            col_step_subi = self.psags.cut_size - self.overlap
            subimgs, sub_positions = moving_window_tiling_ref(self.arr, row_step_subi, col_step_subi, self.psags.cut_size, sort_zero=False, pad=True)

            for sidx, (subimg, subpos) in enumerate(zip(subimgs, sub_positions)):
                # loop through subimages
                print('I am processing sub-image Nr. {} out of {}.'.format(sidx, len(subimgs)))
                # if sidx == 2:
                #     return # for development/debugging
                self.appimg = subimg
                self.apppos = subpos
                self.sidx = sidx
                if np.count_nonzero(subimg) > 0:
                    self.start_time_tiles = time.time()
                    # filter out empty subimages right away
                    self.apply()
                    total_num_tiles.append(self.num_tiles)
        else:
            # run 'apply' for whole array
            self.appimg = self.arr
            self.apppos = (0, 0) # then, the origin is originally (0, 0)
            self.sidx = 0 # simply 0, because there is only one 'subimg'
            self.start_time_tiles = time.time()
            self.apply()
            total_num_tiles.append(self.num_tiles)

        # save full processing time
        time_to_file(self.start_time_full, np.sum(np.array(total_num_tiles)), self.info_path, self.geotiff_filename, -1)

        return
    

class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)


#%%


if __name__ == '__main__':
    '''
        input:
            geofile: the full path of the file for which predictions are seeked. Must be a TIFF file. e.g. '/home/ch19d566/titania/space/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/E6ESCRATER01_GalileoSSI_Equi-cog.tif'

    '''

    # look for input:
    # filter warning from skimage, bool type...
    # warnings.filterwarnings('ignore', '.*Interpolation.*', )


    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect lineaments.')

    parser.add_argument('--geofile', required=True,
                        type=str,
                        metavar="full path with filename in TIFF format",
                        help='the full path of the file for which predictions are seeked. Must be a TIFF file. e.g. /home/ch19d566/titania/space/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/E6ESCRATER01_GalileoSSI_Equi-cog.tif')
    parser.add_argument('--savedir', required=True,
                        type=str,
                        metavar="path to directory where the output is stored.",
                        help='there will be subdirectories automatically generated by the function.')
        
    # optional args
    parser.add_argument('--mask_threshold', required=False,
                        default=0.5,
                        type=float,
                        metavar="threshold for float mask",
                        help='The float mask that is output by the model gets converted to a binary mask using this threshold. Default is 0.5.')
    parser.add_argument('--iou_threshold', required=False,
                        default=0.5,
                        type=float,
                        metavar="threshold for Intersection-Over-Union (IoU)",
                        help='The computed mask IoU is multiplied with the multiplication factor and then tested against the IoU threshold.')
    parser.add_argument('--multiplication_factor', required=False,
                        default=1.0,
                        type=float,
                        metavar="factor by which computed mask Intersection-Over-Union (IoU) gets multiplied.",
                        help='This compensates for the moving window algorithm.')
    parser.add_argument('--del_pxs', required=False,
                        default=40,
                        type=int,
                        metavar="If a boolean mask has an area lower than del_pxs, it gets deleted.",
                        help='If a boolean mask has an area lower than del_pxs, it gets deleted.')
                
        
    # class scores passed as 'dictionary': https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python 
    parser.add_argument('--class_scores', required=False,
                        default=np.array([0.5]),
                        action=Store_as_array,
                        metavar="list with score thresholds for individual classes.",
                        type=float,
                        nargs='*',
                        help='If not given, defaults to 0.5 for each class. Classes are 1) bands 2) double ridges 3) ridge complexes 4) undifferentiated lineae.')
    

    parser.add_argument('--test', required=False,
                        default=False,
                        type=bool,
                        metavar="test mode",
                        help='boolean indicating if we are in test mode (saves preview images, takes excerpt of image ([:550,:550]))')
    parser.add_argument('--plot', required=False,
                        default=False,
                        type=bool,
                        metavar="plotting (only in test mode)",
                        help='boolean indicating if we should plot (e.g. for jupyter notebook) in test mode (saves preview images, takes excerpt of image ([:550,:550]))')
    parser.add_argument('--geosize', required=False,
                        default=224,
                        type=int,
                        metavar="Choose a tile size that is fed to the network.",
                        help='This tile gets re-cast to 200x200 or 300x300. (minsize, maxsize)')
    parser.add_argument('--cut_size', required=False,
                        default=2000,
                        type=int,
                        metavar="Choose a size for cutting the input image into subimages.",
                        help='The image gets tiled up into smaller subimages, if it is bigger than cut_size.')
    parser.add_argument('--azimuth_diff_range', required=False,
                        default=10,
                        type=int,
                        metavar="Choose a maximal difference between two azimuths so that they are still considered the same direction.",
                        help='Masks that fulfill the iou criterion, but are not going into the same direction, are not merged.')
    parser.add_argument('--modelname', required=False,
                        default="./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt",
                        type=str,
                        metavar="path to .pt file of the model.",
                        help='e.g. ./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt')
    parser.add_argument('--minsize', required=False,
                        default=200,
                        type=int,
                        metavar="Choose the minsize parameter for the Mask R-CNN.",
                        help='This tile gets re-cast to (minsize, maxsize)')        
    parser.add_argument('--maxsize', required=False,
                        default=300,
                        type=int,
                        metavar="Choose the maxsize parameter for the Mask R-CNN.",
                        help='This tile gets re-cast to (minsize, maxsize)')        
    parser.add_argument('--sampath', required=False,
                        default="./ckpts/instseg_bb.pt",
                        type=str,
                        metavar=".pt file of the model, in the model_dict subdirectory.",
                        help='e.g. ./ckpts/instseg_bb.pt')        

    # get arguments and put them into 'args' dict
    pargs = parser.parse_args()
    # test with: 
    # pargs = parser.parse_args(["--geofile='/home' --savedir"])
    # pargs = parser.parse_args(["--geofile=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/E6ESCRATER01_GalileoSSI_Equi-cog.tif", "--savedir=z:/Groups/PIG/Caroline/lineament_detection/pytorch_maskrcnn/output/full_imgs/test_LineaMapper_class", "--test=False", "--plot=True"])
    # pargs = parser.parse_args(["--geofile=z:/Groups/PIG/Caroline/isis/data/galileo/usgs_photogrammetrically/Europa_Mosaics_Equirectangular/17ESREGMAP01EXCERPT1_GalileoSSI_Equi-cog.tif",  "--savedir=z:/Groups/PIG/Caroline/lineament_detection/pytorch_maskrcnn/output/full_imgs/test_LineaMapper_class"])
    # pargs = parser.parse_args(["--geofile='/home'"])
    # NOTE: I could not make it run with a class score dict somehow

    # apply LineaMapper to input file
    predMapper = LineaMapper(pargs)
    predMapper.forward()

    # done








# %%
