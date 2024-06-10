# 2024-05-22
# Caroline Haslebacher
# with this script, you can analyse SAM trained on galileo data by Javier Gamazo Tejero

#%%
import h5py
import pandas as pd
import torch

# import the standard data loader
from data_loader import Lineament_dataloader

import os
from pathlib import Path
from datetime import datetime

# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import ImageColor, ImageDraw, ImageFont
import torchvision.transforms.functional as F

import numpy as np
import time

from detection.engine import evaluate_fromoutput


#%%
# define segmentation and mask plotting routine

# define segmentation and mask plotting routine
def segmask_box(img, target, threshold=0.5, score_thresh=0.5, alpha=0.8, colors=None, width=1, fontsize=7, font=None,
                del_pxs=10, del_boxes=True):
    """
        img: torch.tensor or numpy.array, should have shape (H, W, C)
        target: dictionary; output of dataloader
                target['masks'] and target['boxes'] must be defined
        alpha: alpha value for segmentations masks
        colors: specify color (but this is not properly implemented yet)
        width: width to draw bounding box
        del_pxs:  number of pixels a boolean mask must have in order to not be deleted.
        del_boxes: automatically delete empty boxes. Might be unwanted behaviour if mask performance is low.

        returns an image with segmentation masks and bounding boxes drawn on it
    """

    # Handle Grayscale images
    if img.shape[0] == 1:
        img = torch.tile(img, (3, 1, 1))
    # check if shape is correct (H, W, C)
    if not img.shape[0] == 3 or img.shape[0] == 1:
        raise Warning("please help me with the shape. Current shape is: {}, but I expect (H, W, C)".format(img.shape))

    # the image can be a torch tensor with range 0-1 and type float
    # it gets converted to 0-255 uint8
    img = torch.tensor(np.asarray(F.to_pil_image(img)))

    if 'scores' in target.keys():
        # get scores (they are ordered!) and cut off at threshold
        scores = target["scores"]
        cutoff_idc = torch.where(scores > score_thresh)
        # example (tensor([0, 1, 2, 3], device='cuda:0'),)

        # get masks and boxes
        masks = target["masks"][cutoff_idc].cpu()
        boxes = target["boxes"][cutoff_idc].cpu()
        labels = target["labels"][cutoff_idc].cpu()
        scores = target["scores"][cutoff_idc].cpu()

    else:
        # get masks and boxes
        masks = target["masks"].cpu()
        boxes = target["boxes"].cpu()
        labels = target["labels"].cpu()

    out_dtype = torch.uint8
    # define colors
    num_boxes = masks.shape[0]
    if colors is None:
        # colors = torchvision.utils._generate_color_palette(num_boxes)
        # actually, get colors depending on the label
        # bands, 1 --> darkseagreen
        # double ridges, 2 --> maroon
        # ridge complexes, 3 --> royalblue
        # undifferentiated linea, 4 --> olive
        # color_dict = {0: 'black', 1: 'darkseagreen', 2: 'maroon', 3: 'royalblue', 4: 'olive' }
        # color_dict = {0: 'black', 1: 'green', 2: 'maroon', 3: 'deepskyblue', 4: 'khaki' }
        # new: (revision 1)
        # color_dict = {0: 'black', 1: '#7F4A9D', 2: '#ED9A22', 3: '#FFD380', 4: '#D9B6D6', 5: '#D9B6D7' }
        # pub2 (regiomaps):
        color_dict = {0: 'black', 1: '#7F4A9D', 2: '#ED9A22', 3: '#8ED311', 4: '#00FFC5', 5: '#00FFC5'}
        # new colordict from qgis: ... not convinced
        # color_dict = {0: 'black', 1: '#f7fbff', 2: '#cde0f2', 3: '#98c7e0', 4: '#1c6bb1'}
        colors = [color_dict[label.item()] for label in labels]

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        colors_.append(torch.tensor(color, dtype=out_dtype))

    # convert image to numpy array
    # ndarr = img.permute(2,0,1).cpu().numpy()
    # convert bounding boxes to list
    img_boxes = boxes.to(torch.int64).tolist()

    img_to_draw = img.detach().clone()  # clone img to draw on it

    del_idx = []

    # TODO: There might be a way to vectorize this
    for i, (mask, color) in enumerate(zip(masks, colors_)):
        # print(i)
        # boolean indexing
        if threshold != None:
            # for model predictions, we need threshholding!
            mask = mask[0]
            mask[mask < threshold] = 0  # mask is currently between 0 and 1

        # if mask is completely empty after thresholding,
        # get rid of corresponding bounding box
        # in fact, get rid of all bounding boxes that contain only a very small amount of pixels
        if torch.count_nonzero(mask).item() <= del_pxs:
            # print('empty mask detected')
            # plt.imshow(mask)
            # plt.show()
            del_idx.append(i)

        else:  # else, we draw the mask
            img_to_draw[mask.bool()] = color[None, :]

        # img_to_draw[mask.bool()] = color # shape (H,W,C)

    if del_boxes:
        # delete boxes and labels and scores and colors that are in del_idx
        # quickly convert to array
        temp = np.array(img_boxes)
        templ = np.array(labels)
        tempc = np.array(colors)
        # print(del_idx)
        # print(temp.shape)
        tempb = np.delete(temp, del_idx, axis=0)
        templ = np.delete(templ, del_idx, axis=0)
        tempc = np.delete(tempc, del_idx, axis=0)
        if 'scores' in target.keys():
            temps = np.array(scores)
            temps = np.delete(temps, del_idx, axis=0)
            scores = temps.tolist()
        # print(tempb.shape)
        img_boxes = tempb.tolist()
        labels = templ.tolist()
        colors = tempc.tolist()

    else:
        # else, we display empty (no masks) boxes as well
        # convert to array
        img_boxes = np.array(img_boxes).tolist()
        labels = np.array(labels).tolist()
        colors = np.array(colors).tolist()

    out = img * (1 - alpha) + img_to_draw * alpha
    out = out.to(out_dtype)
    # plt.imshow(out)
    # plt.show()

    # save this mask image alone
    out_masks = out.clone()

    # bounding boxes
    img_for_bbox = F.to_pil_image(out.permute(2, 0, 1))
    draw = ImageDraw.Draw(img_for_bbox)  # Image.fromarray(out.cpu().numpy())

    txt_font = ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=fontsize)
    # txt_font = ImageFont.truetype("LiberationSansNarrow-Regular.ttf", size=fontsize)
    draw.fontmode = "L"
    # ImageFont.load_default()

    # here, we need colors in int or tuple format
    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            # get category from class dict
            margin = width + 1
            try:
                draw.text((bbox[0] + margin, bbox[1] + margin), str(label), fill=(0, 0, 0),
                          font=txt_font)  # text in black
                # or print cat_dict[label.item()] instead of label
            # note: to make text white, just get rid of fill=color
            except KeyError:
                # then we have an undefined category
                undef_label = 'undefined'
                draw.text((bbox[0] + margin, bbox[1] + margin), undef_label, fill=(0, 0, 0), font=txt_font)
    result = torch.from_numpy(np.array(img_for_bbox)).permute(2, 0, 1).to(dtype=torch.uint8)
    # plt.imshow(result.permute(1,2,0))
    # plt.show()

    return result, out_masks


def main(config, datafolders, shiftbools, rcnnsam_args):
    '''
    for debugging, use:
        run = runs[0]
        model_name = model_names[0]
        datafolder = datafolders112[1]
    '''
    for run, model_name in zip(config['runs'], config['model_names']):
        for datafolder, shift5to4 in zip(datafolders, shiftbools):

            figsubset = run + '_' + model_name

            # following loop creates folders with predictions on the test set
            # and evaluates metrics on them if chosen (these get saved in the home directory, because I needed to
            # implement it right in ...cocoeval.py source code)

            # already here, get the folder path for saving
            now = datetime.now()
            dt_ymd = now.strftime("%Y_%m_%d_%H_%M_%S_")

            ####
            #  visualize predictions
            # pick one image from the test set
            # train on the GPU or on the CPU, if a GPU is not available
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            # automatically look inside this hard-coded folder for the datafolder subfolder
            mygalipath = basepath / 'data/tiles' / datafolder
            # validation set
            dataset = Lineament_dataloader(mygalipath / 'test', transform=None, bbox_threshold=20, shift5to4=shift5to4)

            # get class_dict
            class_dict = dataset.load_class_dict()
            # cat_dict = {v: k for k, v in dataset.load_class_dict().items()}
            # num_classes = len(class_dict) + 1

            # get the model using our helper function
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

            batch_size = 6
            dataloader_val = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x)))
            # # check output:
            # images, targets = next(iter(dataset))
            # images = list(image for image in images)
            # targets = [{k: v for k, v in t.items()} for t in targets]
            # output = model(images,targets)   # Returns losses and detections
            # rcnnSAM
            mask_generator, box_generator = get_rcnnSAM(rcnnsam_args)

            model_output = []

            # PLOT
            start_time = time.time()
            for idx in range(len(dataset)): # len(dataset)
                img, target = dataset[idx]
                # # include a shift5to4-filter here!
                # # loop through targets['labels]
                # del_idcs = []
                # for laidx in range(0, len(target['labels'])):
                #     if target['labels'][laidx] == 4:
                #         print('deleting 4')
                #         target['labels'] 
                #         del_idcs.append(laidx)
                #     if target['labels'][laidx] == 5:
                #         print('shifting to 4')
                #         target['labels'][laidx] = 4
                # now delete in reverse order, because it's lists

                result = get_rcnnsam_output(mask_generator, box_generator, img)
                model_output.append(result[0])

                # and retrieve prediction 
                prediction = model_output[idx]
                # print("--- %s seconds for this ----" % (time.time() - start_time))
                # time needed: 1.53 seconds for 11 images, or 1.48 s for 10 images
                # prediction is like 'target'
                # use segmask function from data_viewer

                # index prediction[0] to get rid of batch dimension
                t, masks = segmask_box(img, prediction, threshold=config['mask_threshold'],
                                       score_thresh=config['score_thresh'], alpha=0.8, colors=None, width=1, fontsize=6,
                                       font=None, del_boxes=False)
                # ground truth:
                tgr, masks_gr = segmask_box(img, target, threshold=None, alpha=0.8, colors=None, width=1, fontsize=6,
                                            font=None, del_boxes=False)
                # font="LiberationSansNarrow-Regular.ttf"

                fig, ax = plt.subplots(ncols=3, figsize=(20, 40))  # ncols=4
                # raw image
                ax[0].imshow(img.permute(1, 2, 0))
                # ax[0].set_title('input image')
                # mask
                # ax[1].imshow(masks)
                # masks and bounding boxes
                # ax[1].set_title('model prediction')
                ax[1].imshow(t.moveaxis(0, 2))  # for EGU, don't plot bboxes because there are too many..
                # ground truth
                # ax[2].set_title('ground truth')
                ax[2].imshow(tgr.moveaxis(0, 2))
                # individuals
                # plot img
                imgimg = F.to_pil_image(img)
                # plot predictions
                imgpreds = F.to_pil_image(t)
                # plot ground truth
                imggr = F.to_pil_image(tgr)
                # all saved below

                # title = img_path
                # save fig to output
                # give a specific name
                if config['save']:
                    # pdf folder
                    if config['pdf']:
                        save_path_pdf = basepath / 'output/maskrcnn' / (dt_ymd + run) / 'pdf'
                        fig_path = save_path_pdf.joinpath(
                            target['path'].stem + '_' + figsubset + '_' + str(idx) + run + '_score_thr_' + str(config['score_thresh']) + '_mask_thr_' + str(config['mask_threshold']) + '.pdf'
                        )
                        os.makedirs(save_path_pdf, exist_ok=True)
                        fig.savefig(fig_path, bbox_inches='tight', facecolor='white')


                    # png folder
                    if config['png']:
                        save_path_png = basepath / 'output/maskrcnn' / (dt_ymd + run) / 'png'
                        fig_path = save_path_png.joinpath(f'{target["path"].stem}_{figsubset}_{str(idx)}{run}.png')
                        os.makedirs(save_path_png, exist_ok=True)
                        fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
                        # to make it shorter: '_threshold_' + str(score_thresh) + '_mask_thr_' + str(mask_threshold) +

                    # save individual files
                    indpath = basepath / 'output/maskrcnn' / (dt_ymd + run) / 'png' / 'individuals'
                    os.makedirs(indpath, exist_ok=True)
                    imgimg.save(indpath.joinpath(figsubset + str(idx) + run + '_img.png'), quality=95)
                    imgpreds.save(indpath.joinpath(figsubset + str(idx) + run + '_prediction.png'), quality=95)
                    imggr.save(indpath.joinpath(figsubset + str(idx) + run + '_groundtruth.png'), quality=95)
                    plt.close('all')
                # #  note: output for several
                # NEW: get model output of SAM in correct format:
                # model_output = hdf_to_output(hdfdatafolder, dataloader_val)  #

                # check length:
                print(f'length of dataset is {len(dataset)}, length of model_output is {len(model_output)}')

            # after model_output is complete
            # EVALUATE
            if config['evaluate_now']:
                # I've changed 'evaluate' to output individual class APs
                mdict = evaluate_fromoutput(model_output, dataloader_val, device=device, savepath='./metrics')
                mdict['map']  # this is the total map
                mdict['map_per_class']  # this is a tensor
                # CAUTION: I modified 'evaluate' so that mdict gets stored in C:\Users\ch20s351, from where I can get it



#%%
if __name__ == '__main__':
    current = os.getcwd()
    titaniach = Path(current.split('Caroline')[0]) / 'Caroline'

    # import modules from folder above:
    import sys
    sys.path.append((titaniach / 'lineament_detection/Reinforcement_Learning_SAM/europa_surface').as_posix())
    from LineaMapper_v2_to_img import get_sam_model, load_LineaMapper, get_rcnnsam_output, get_rcnnSAM, batch_process_rcnnsam

    # import sys
    # sys.path.append((titaniach / 'lineament_detection/Reinforcement_Learning_SAM/europa_surface/src/performance/detection').as_posix())
    # from detection.engine import evaluate_fromoutput


    basepath = titaniach / 'lineament_detection/RegionalMaps_CH_NT_EJL_LP/mapping/'

    def run_main(datafolders, shiftbools, img_size):
        '''
            for changes in datafolder coupled to img_size
        '''

        model_names = [
            'rcnnSAM_v1',   # LineaMapper v1.0
        ]
        runs = ['run01']
        # hdfdatafolder = Path('Z:\Groups\PIG\Caroline\lineament_detection\Reinforcement_Learning_SAM/test_SAMv1')

        ckpts_path = titaniach / 'lineament_detection/Reinforcement_Learning_SAM/europa_surface/ckpts'
        rcnnsam_args = {
                'num_classes': 5,
                'img_size': img_size, # update to 112x112 dataset
                'ckpt_path': ckpts_path.joinpath('instseg_bb.pt'),
                'model_name': ckpts_path.joinpath("Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt"),
                'minsize': 200,
                'maxsize': 300, 
                'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            }

        config = {
            'runs': runs,
            'model_names': model_names,
            'save': True,
            'evaluate_now': True,
            'mask_threshold': 0.5,  # should be 0.5 to be consistent with metrics...
            'score_thresh': 0.5,  # FOR SAM
            'png': True,
            'pdf': False,
        }
        # run main
        main(config, datafolders, shiftbools, rcnnsam_args)

        return



    datafolders224 = [
        # "2024_02_14_11_22_Regiomaps_112x112",
        "2023_10_16_15_34_Regiomaps_224x224",  # correct comparison for SAM
        "val_and_train_pytorch_224x224_LINEAMAPPER_v1.0", # old test set, 224x224
        # "val_and_train_pytorch_112x112_LINEAMAPPER_v1.0", # old test set, re-tiled to 112x112
    ]
    shiftbools = [False, True] # false for regiomaps, true vor v1.0

    run_main(datafolders224, shiftbools, 224)

    datafolders112 = [
        "2024_02_14_11_22_Regiomaps_112x112",
        # "2023_10_16_15_34_Regiomaps_224x224",  # correct comparison for SAM
        # "val_and_train_pytorch_224x224_LINEAMAPPER_v1.0", # old test set, 224x224
        "val_and_train_pytorch_112x112_LINEAMAPPER_v1.0", # old test set, re-tiled to 112x112
    ]
    shiftbools = [False, True] # false for regiomaps, true vor v1.0
    
    run_main(datafolders112, shiftbools, 112)
    

#%%