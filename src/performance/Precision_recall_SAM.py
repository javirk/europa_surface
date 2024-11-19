# 2023-02-22
# Caroline Haslebacher
# calculation of the precision and the recall separately
# copied from \\titania.unibe.ch\Space\Groups\PIG\Caroline\lineament_detection\pytorch_maskrcnn\code\galileo_dataset_pub2_nocusps\pub_files\code
# ATTENTION: this imports from edited coco_eval.py and coco_utils.py in './detection' (from 'vision')

#%%
import torch
# import the data loader
from data_loader import Lineament_dataloader

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import numpy as np
import time
import pandas as pd

import h5py

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# ATTENTION: use pip install "git+https://github.com/CarolineHaslebacher/cocoapi.git@cocoeval_for_multiple_categories#egg=pycocotools&subdirectory=PythonAPI"
# in order to apply changes
from detection.coco_eval import CocoEvaluator
from detection.coco_utils import get_coco_api_from_dataset



def hdf_to_output(hdfpaths_orig, dataloader_val):
    '''
    first, load the hdf5 file !! for each image_path in dataloader_val !! to match
    then, convert to tensor
    output as list of dictionary
    '''
    outputs = []

    # we loop through items in dataloader_val. the length depends on the batch size (usually 6)
    for _, targets in iter(dataloader_val):
        for target in targets:
            # for each target, we get the corresponding hdf5 file, load it and save it in order
            tarpath = target['path']
            # print(tarpath)

            # get path stem
            hdfpath = hdfpaths_orig.joinpath(tarpath.stem + '.hdf5')

            # check if path exists:
            if hdfpath not in sorted(hdfpaths_orig.glob('*.hdf5')):
                # add empty prediction and continue
                # to keep order!
                # convert everything into a torch.Tensor
                boxes = torch.as_tensor([[1.0, 1.0, 4.0, 4.0]],
                                        dtype=torch.float32)  # [None] # add a dimension since model expects tensor of shape [N, 4]
                # there are multiple classes
                labels = torch.as_tensor([0], dtype=torch.int64)
                # masks as tensor
                masks = torch.as_tensor(np.zeros((1, 1, target['masks'].shape[1], target['masks'].shape[2])),
                                        dtype=torch.uint8)
                scores = torch.as_tensor([0], dtype=torch.int64)

            else:  # load hdf5 results
                # load hdf5
                with h5py.File(hdfpath, "r") as f:
                    masks = f['mask'][:]  # (10, 224, 224), but needs to be (10, 1, 224, 224)
                    n_mask = masks.shape[0]
                    labels = f['labels'][:]  # (10,)
                    n_labels = labels.shape[0]
                    # logits_mask = f['logits_mask'][:] # (10, 224, 224)
                    boxes = f['bbox'][:]  # (10,4)
                    n_boxes = boxes.shape[0]
                    scores = f['scores'][:]  # (10,) IDEALLY, SCORES WOULD BE IN ORDER FOR CUTOFF.
                    # print(scores)
                    n_scores = scores.shape[0]

                # sort by scores: how? dataframe and numpy.sort both do not work off the shelve.

                # check number of predictions, has to be equal!
                # print(len(np.unique(np.array([n_mask, n_boxes, n_labels, n_scores]))))
                if len(np.unique(np.array([n_mask, n_boxes, n_labels, n_scores]))) != 1:
                    print('this one is not equal: {}'.format(hdfpath))

                '''
                HDF5 contain all the data that you need (I believe) to compute AP. Each file has the data for an image and contains 5 fields:
                “mask”: Boolean, thresholded mask [N, H, W], where N is the number of instances.
                “labels”: [N]. Class ID for each instance
                “logits_mask”: [N, H, W] logits mask (before sigmoid)
                “bbox”: [N, 4] Bounding boxes in (Xmin Ymin Xmax Ymax) format
                “scores”: [N] Score for each instance. This is actually the IoU prediction from the model and, to be honest, not completely reliable

                '''

                # convert everything into a torch.Tensor
                boxes = torch.as_tensor(boxes,
                                        dtype=torch.float32)  # [None] # add a dimension since model expects tensor of shape [N, 4]
                # there are multiple classes
                labels = torch.as_tensor(labels, dtype=torch.int64)
                # masks as tensor
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                # add one dimension
                masks = masks[:, None, :, :]
                # scores as tensor
                scores = torch.as_tensor(scores, dtype=torch.float32)

            model_output = {}
            model_output["boxes"] = boxes
            model_output["labels"] = labels
            model_output["masks"] = masks
            model_output['scores'] = scores
            model_output['corrpath'] = tarpath  # corresponding path for cross-checkes

            outputs.append(model_output)

    return outputs  # as list, like mask R-CNN original output


# define recall thresholds:
rec_thrs = np.array([0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                     0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,
                     0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32,
                     0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43,
                     0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54,
                     0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65,
                     0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
                     0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
                     0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
                     0.99, 1.])


def main(root, data_root, datafolders, shiftbools, hdfdatafolder, save_path, subset='test'):
    if isinstance(hdfdatafolder, str):
        hdfdatafolder = Path(hdfdatafolder)
    for datafolder, shift5to4 in zip(datafolders, shiftbools):
        # automatically look inside this hard-coded folder for the datafolder subfolder
        mygalipath = data_root / datafolder
        # TEST SET
        dataset = Lineament_dataloader(mygalipath / subset, transform=None, bbox_threshold=20, shift5to4=shift5to4)

        batch_size = 6
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: list(zip(*x))
        )

        # get class_dict
        class_dict = dataset.load_class_dict()
        cat_dict = {v: k for k, v in dataset.load_class_dict().items()}
        num_classes = len(class_dict) + 1

        ##### create cocogt and cocomodel
        # from 'engine.py'
        iou_types = ['bbox', 'segm']
        # ground truth
        coco = get_coco_api_from_dataset(dataloader.dataset)  # this returns an object <pycocotools.coco.COCO>
        coco_evaluator = CocoEvaluator(coco, iou_types)

        # load full model output (we load only the files in the dataloader, in the correct order)
        model_output = hdf_to_output(hdfdatafolder, dataloader)

        # from engine.py
        for imt_idx, (images, targets) in enumerate(dataloader):
            # put images to list
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # get model predictions
            outputs = model_output[imt_idx * batch_size: imt_idx * batch_size + len(images)]
            # for target, output in zip(targets, outputs):
            #     print(target['path'].stem)
            #     print(output['corrpath'].stem)
            #     print(target['path'].stem == output['corrpath'].stem)

            # put outputs into good format, and to CPU
            # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)

        # now we can evaluate 
        coco_evaluator.synchronize_between_processes()  # I don't know what this does, but it is crucial
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        # here is something wrong...

        # now I can access the .eval dictionary, where the precision and recall is stored!
        #         self.eval = {
        #     'params': p,
        #     'counts': [T, R, K, A, M],
        #     'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #     'precision': precision,
        #     'recall':   recall,
        #     'scores': scores,
        # }
        coco_evaluator.coco_eval['bbox'].eval
        coco_evaluator.coco_eval['segm'].eval

        # shape: coco_evaluator.coco_eval['bbox'].eval['precision'].shape 
        # = (10, 101, 4, 4, 3)
        # format: T,R,K,A,M (10 IoU thresholds, 101 recall thresholds, 4 categories, 4 area Ranges, 3 Max Detection numbers)
        # T           = len(p.iouThrs[1:]) # array([0.35, 0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95]) (with coco_evaluator.coco_eval['bbox'].eval['params'].iouThrs)
        # R           = len(p.recThrs) # coco_evaluator.coco_eval['bbox'].eval['params'].recThrs,  array starting at 0 and ending at 1 in steps of 0.01
        # K           = len(p.catIds) if p.useCats else 1
        # A           = len(p.areaRng) # accessed with coco_evaluator.coco_eval['bbox'].eval['params'].areaRng --> [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
        # M           = len(p.maxDets) # usually [1 10 100]

        # the scores have the same shape: 
        # coco_evaluator.coco_eval['bbox'].eval['scores'].shape = (10, 101, 4, 4, 3)

        # for stating the precision in the publication, we want 
        # T: IoU threshold of 0.5 --> index 1
        # K: all categories --> index :
        # A: 'all' areas --> index 0
        # score >= 0.5 (and corresponding recalls)
        # M: 100 Maxdets --> index 2

        #############
        # plotting

        # old: color_dict = {0: 'black', 1: 'green', 2: 'maroon', 3: 'deepskyblue', 4: 'khaki' }
        # update for revision:
        color_dict = {0: 'black', 1: '#7F4A9D', 2: '#ED9A22', 3: '#8ED311', 4: '#00FFC5', 5: '#00FFC5'} # {0: 'black', 1: '#7F4A9D', 2: '#ED9A22', 3: '#FFD380', 4: '#D9B6D6'}
        # create custom lines for legend
        linewid = 2
        custom_lines = [Line2D([0], [0], color=color_dict[1], lw=linewid),
                        Line2D([0], [0], color=color_dict[2], lw=linewid),
                        Line2D([0], [0], color=color_dict[3], lw=linewid),
                        Line2D([0], [0], color=color_dict[4], lw=linewid),
                        Line2D([0], [0], color='r', lw=2)]

        # for pandas dataframe, csv
        masks_prc = []
        bbox_prc = []
        masks_rec = []
        bbox_rec = []

        for ioutype in iou_types:
            precision = coco_evaluator.coco_eval[ioutype].eval['precision']  # (10, 101, 5, 4, 3)
            recall = coco_evaluator.coco_eval[ioutype].eval[
                'recall']  # SAM: (10, 5, 4, 3), Mask R-CNN was of shape (10, 4, 4, 3)
            scores = coco_evaluator.coco_eval[ioutype].eval[
                'scores']  # SAM: (10, 101, 5, 4, 3), Mask R-CNN was of shape (10, 101, 4, 4, 3)
            t_ind = 1  # t_ind=0--> IoU=0.35, t_ind=1 --> IoU=0.5
            # or IoU threshold index = 0 if iou = 0.35
            a_ind = 0  # 0 --> 'all'
            m_ind = 2  # 2 --> 100 max detections
            precision = precision[t_ind, :, :, a_ind, m_ind]  # now the shape is (101, 5)
            scores = scores[t_ind, :, :, a_ind, m_ind]  # now the shape is (101, 5)
            recall = recall[t_ind, :, a_ind, m_ind]  # has shape (5,), average recall at IoU=0.5 

            fig, axs = plt.subplots(1, 1, figsize=(7, 5))
            # segs = [np.column_stack([rec_thrs, precision[:,cati]]) for cati in range(4)]

            # # Create a continuous norm to map from data points to colors
            # norm = plt.Normalize(dydx.min(), dydx.max())
            # lc = LineCollection(segments, cmap='viridis', norm=norm)
            # # Set the values used for colormapping
            # lc.set_array(dydx)
            # lc.set_linewidth(2)
            # line = axs[0].add_collection(lc)
            # fig.colorbar(line, ax=axs[0])
            # # from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html 

            # precision-recall curve for all categories is now straight-forward
            # Iou-Type Dictionary
            itd = {'bbox': 'bounding boxes', 'segm': 'masks'}
            # Category Dictionary
            cd = {'0': 'bands', '1': 'double ridges', '2': 'ridge complexes', '3': 'undiff. lineae'}
            for cat_idx in range(4):  # changed back to 4. this was a mistake before implementation of shift5to4
                points = np.array([rec_thrs, precision[:, cat_idx]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # prcurv = axs.plot(rec_thrs, precision[:,cat_idx], label=cd[str(cat_idx)])
                cmap = ListedColormap(['r', color_dict[cat_idx + 1]])
                norm = BoundaryNorm([0, 0.5, 1], cmap.N)  # old: plt.Normalize(0, 1) (no difference!)
                lc = LineCollection(segments, cmap=cmap, norm=norm, joinstyle='bevel',
                                    linewidth=linewid)  # , norm=norm, cmap='viridis'
                lc.set_array(scores[1:, cat_idx])  # --> number of segments: 100, number of scores: 101 --> I need to 
                # lc.set_linewidth(linewid)
                line = axs.add_collection(lc)
                # fig.colorbar(line, ax=axs)

                # output precision and recall for score >= 0.5
                # np.take(precision[:,cat_idx], scores[:,cat_idx]>=0.5)
                if (scores[:, cat_idx] >= 0.5).any():  # same as if I would put: == True
                    print('precision at score of 0.5 and IoU=0.5: {:.3f}'.format(
                        precision[:, cat_idx][scores[:, cat_idx] >= 0.5].min()))
                    print(
                        'recall at score of 0.5 and IoU=0.5: {:.3f}'.format(rec_thrs[scores[:, cat_idx] >= 0.5].max()))
                    if ioutype == 'bbox':
                        bbox_prc.append('{:.2f}'.format(precision[:, cat_idx][scores[:, cat_idx] >= 0.5].min()))
                        bbox_rec.append(rec_thrs[scores[:, cat_idx] >= 0.5].max())
                    if ioutype == 'segm':
                        masks_prc.append('{:.2f}'.format(precision[:, cat_idx][scores[:, cat_idx] >= 0.5].min()))
                        masks_rec.append(rec_thrs[scores[:, cat_idx] >= 0.5].max())
                else:
                    if ioutype == 'bbox':
                        bbox_prc.append('0.00')
                        bbox_rec.append('0.00')
                    if ioutype == 'segm':
                        masks_prc.append('0.00')
                        masks_rec.append('0.00')
            # axs.vlines(0.54, 0, 1)
            # axs.legend()
            # axs.set_title('precision-recall curve for {}'.format(itd[ioutype]))
            axs.set_aspect('equal')
            axs.set_xlabel('recall')
            axs.set_ylabel('precision')
            axs.set_xticks(np.arange(0, 1.1, 0.1))
            axs.set_yticks(np.arange(0, 1.1, 0.1))
            axs.set_ylim(0, 1.05)
            axs.legend(custom_lines, list(cd.values()) + [r'score $\leq$ 0.5'])
            fig_path = (root / save_path).joinpath(
                f'precision-recall_curve_T{t_ind}_A{a_ind}_M{m_ind}_for_{itd[ioutype]}_{subset}_{datafolder}.pdf'
            )
            fig.savefig(fig_path)
            fig.show()

        # precision and recall at score of 0.5 and IoU threshhold of 0.5
        # to dataframe
        df = pd.DataFrame([
            np.array(bbox_prc, dtype='float64'), bbox_rec, np.array(masks_prc, dtype='float64'), masks_rec]).T
        df = df.apply(pd.to_numeric)
        df.columns = ['bbox_precision', 'bbox_recall', 'masks_precision', 'masks_recall']
        # add 'average' row by calculating the mean along the axis 0
        df.loc[len(df.index)] = df.mean(axis=0)
        # add indices
        df.index = ['bands', 'double ridges', 'ridge complexes', 'undiff. lineae', 'average']
        # save as csv
        csv_path = (root / save_path).joinpath(
            f'precision_recall_at_score0.5_IoU0.5__T{t_ind}_A{a_ind}_M{m_ind}_{subset}_{datafolder}.csv'
        )
        df.to_csv(csv_path)


def get_caroline_config():
    current = os.getcwd()
    root = Path(current.split('Caroline')[0]) / 'Caroline'
    datafolders = [
        # "2024_02_14_11_22_Regiomaps_112x112",
        "2023_10_16_15_34_Regiomaps_224x224",  # correct comparison for SAM
        "val_and_train_pytorch_224x224_LINEAMAPPER_v1.0",  # old test set, 224x224
        # "val_and_train_pytorch_112x112_LINEAMAPPER_v1.0", # old test set, re-tiled to 112x112
    ]
    shiftbools = [False, True] # false for regiomaps v1.1 (new dataset), true vor v1.0 (old dataset)
    save_path = 'lineament_detection/RegionalMaps_CH_NT_EJL_LP/mapping/output/precision_recall/SAM_v1'
    #  load model and data
    data_root = root / 'lineament_detection/RegionalMaps_CH_NT_EJL_LP/mapping/data/tiles'
    hdfdatafolder = root / 'lineament_detection/Reinforcement_Learning_SAM/test_SAMv1'
    subset = 'test'

    return root, data_root, datafolders, hdfdatafolder, save_path, subset, shiftbools


def get_javier_config(hdfdatafolder):
    root = Path(os.getcwd())
    datafolders = [
        "newdataset_224x224",  # new dataset, 224x224
        "dataset_224x224",  # old test set, 224x224
    ]
    shiftbools = [False, True] # false for regiomaps v1.1 (new dataset), true vor v1.0 (old dataset)
    save_path = os.path.join(hdfdatafolder, 'precision_recall')
    #  load model and data
    data_root = Path('/Users/javier/Documents/datasets/europa')
    subset = 'test_raw'

    return root, data_root, datafolders, hdfdatafolder, save_path, subset, shiftbools

#%%
if __name__ == '__main__':
    user = 'caroline'
    if user == 'javier':
        hdf_folder = './results/Galileo/Galileo_20240531-102944'
        root, data_root, datafolders, hdfdatafolder, save_path, subset, shiftbools = get_javier_config(hdf_folder)
    elif user == 'caroline':
        root, data_root, datafolders, hdfdatafolder, save_path, subset, shiftbools = get_caroline_config()
    else:
        raise NotImplementedError

    os.makedirs(root / save_path, exist_ok=True)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(root, data_root, datafolders, shiftbools, hdfdatafolder, save_path, subset=subset)

# %%
