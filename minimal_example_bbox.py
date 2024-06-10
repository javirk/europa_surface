
#%%

import numpy as np

from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.datasets import VanillaGalileoDataset
from LineaMapper_v2_to_img import load_LineaMapper, get_sam_model # , get_rcnnSAM, get_rcnnsam_output

#%%

def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy

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
    box_generator = load_LineaMapper(rcnnsam_args['model_name'], rcnnsam_args['minsize'], rcnnsam_args['maxsize'], rcnnsam_args['device'])

    return mask_generator, box_generator

def get_rcnnsam_output(mask_generator, box_generator, image_formaskrcnn):
    '''
    image: torch tensor (float), shape (C, H, W) (for mask r-cnn)
    '''
    # get mask r-cnn result
    result = box_generator([image_formaskrcnn])
    boxes = result[0]['boxes']

    # prompt sam with mask r-cnn boxes (faster r-cnn)
    image_forsam = (255*image_formaskrcnn).permute(1,2,0).numpy().astype('uint8') # for SAM, input is (H, W, C)
    sam_result = mask_generator.generate(image_forsam, boxes)

    # update result with sam masks
    masks = torch.as_tensor(sam_result['segmentation'][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension
    result[0]['masks'] = torch.as_tensor(masks, dtype=torch.uint8)
    # return the updated result
    return result


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
        # images_forsam = [(255*im).permute(1,2,0) for im in images_formaskrcnn] # for SAM, input is (H, W, C)
        # sam_result = mask_generator.generate(torch.stack(images_forsam), torch.stack(boxes))
        image_forsam = (255*im).permute(1,2,0).numpy().astype('uint8') # for SAM, input is (H, W, C)
        sam_result = mask_generator.generate(image_forsam, result[res_idx]['boxes'])

        # update result with sam masks
        # instead of 'segmentation', which is the thresholded mask, we can take 'logits_mask'
        masks = torch.as_tensor(sam_result['segmentation'][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension
        result[res_idx]['masks'] = torch.as_tensor(masks, dtype=torch.float32)
        # masks.shape: torch.Size([100, 1, 224, 224])
        # masks.dtype: torch.float32        
        
    # return the updated result
    return result


#%%
if __name__ == '__main__':
    dataset_folder = '../datasets_2024-05-31'
    dataset = VanillaGalileoDataset(dataset_folder, 'train')

    ### TODO: replace with function
    # sam = sam_model_registry['vit_b'](checkpoint='./ckpts/instseg_bb.pt', num_classes=5, image_size=224)
    # mask_generator = SamBBoxMaskGenerator(sam)
    # num_classes = 5 
    # img_size = 224
    # ckpt_path = './ckpts/instseg_bb.pt'



    # ### TODO: also load LineaMapper
    # model_name = "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt"
    # minsize = 200
    # maxsize = 300
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # box_generator = load_LineaMapper(model_name, minsize, maxsize, device)


    # load an image
    image_forsam = dataset[0]['image'].permute(1,2,0).numpy() # for SAM, input is (H, W, C)
    image_formaskrcnn = dataset[0]['image']/255

    # # instead, next line # boxes = dataset[0]['bboxes']  # [N, 4]
    # ### TODO: get bbox from LineaMapper, feed into SAM
    # # IN WHICH FORMAT ARE BOXES? xyxy, all good.
    # result = box_generator([image_formaskrcnn])
    # boxes = result[0]['boxes']
    # # and also take labels if you want, to filter:
    # scores = result[0]['scores']


    # # generating the masks from bounding box inputs
    # sam_result = mask_generator.generate(image_forsam, boxes)

    # # The result is a dict with the following keys:
    # # - 'segmentation': np.ndarray with the mask
    # # - 'bbox': np.ndarray with the bounding box in xywh format
    # # - 'predicted_iou': float with the predicted iou

    rcnnsam_args = {
            'num_classes': 5,
            'img_size': 224,
            'ckpt_path': './ckpts/instseg_bb.pt',
            'model_name': "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run08_end_model.pt",
            'minsize': 200,
            'maxsize': 300, 
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        }
    mask_generator, box_generator = get_rcnnSAM(rcnnsam_args)
    result = get_rcnnsam_output(mask_generator, box_generator, image_formaskrcnn)

    # instances = (sam_result['segmentation'][:, 1:].sum(axis=1) > 0) # of shape (N, 224, 224) with N number of masks

    instances = result[0]['masks'][:,0,:,:].numpy().astype(bool) # for visualization, needs to be boolean
    boxes = result[0]['boxes']

    # The rest is for visualization. Completely optional
    import supervision as sv

    box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    instance_detections = sv.Detections(xyxy=boxes.detach().numpy(), mask=instances, class_id=np.zeros(len(boxes)))
    annotated_image_semantic = mask_annotator.annotate(scene=image_forsam.copy(), detections=instance_detections)
    inp = box_annotator.annotate(scene=image_forsam.copy(), detections=instance_detections)

    sv.plot_images_grid(
        images=[inp, annotated_image_semantic],
        grid_size=(1, 2),
        titles=['source image', 'instance segmentation'],
    )

    #######################################################
    # test batch processing:
    images_formaskrcnn = [image_formaskrcnn, image_formaskrcnn]
    result = batch_process_rcnnsam(mask_generator, box_generator, images_formaskrcnn)

    instances = result[1]['masks'][:,0,:,:].numpy().astype(bool) # for visualization, needs to be boolean
    boxes = result[1]['boxes']

    # The rest is for visualization. Completely optional
    import supervision as sv

    box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    instance_detections = sv.Detections(xyxy=boxes.detach().numpy(), mask=instances, class_id=np.zeros(len(boxes)))
    annotated_image_semantic = mask_annotator.annotate(scene=image_forsam.copy(), detections=instance_detections)
    inp = box_annotator.annotate(scene=image_forsam.copy(), detections=instance_detections)

    sv.plot_images_grid(
        images=[inp, annotated_image_semantic],
        grid_size=(1, 2),
        titles=['source image', 'instance segmentation'],
    )
