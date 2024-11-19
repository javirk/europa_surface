
#%%
# @Caroline: use venv vgeosam312
import numpy as np
import torch

from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.datasets import VanillaGalileoDataset
from LineaMapper_v2_to_img import load_LineaMapper, get_sam_model, get_samPoints # , get_rcnnSAM, get_rcnnsam_output
from src.datasets.dataset_utils import ListCollate

from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

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
    mask_generator = get_sam_model(rcnnsam_args)
    # send to device
    mask_generator.predictor.model.to(rcnnsam_args['device'])
    box_generator = load_LineaMapper(rcnnsam_args['model_name'], rcnnsam_args['num_classes'], rcnnsam_args['minsize'], rcnnsam_args['maxsize'], rcnnsam_args['device'])

    return mask_generator, box_generator

####
def get_rcnnsam_output(mask_generator, box_generator, image_formaskrcnn):
    '''
    image: torch tensor (float), shape (C, H, W) (for mask r-cnn)
    image gets re-shaped to SAM directly within this function call
    '''
    # get mask r-cnn result
    result = box_generator([image_formaskrcnn])
    boxes = result[0]['boxes']

    if len(boxes) == 0:
        # then, we simply return the result. this happens if we have an empty images
        return result

    # prompt sam with mask r-cnn boxes (equals faster r-cnn)
    image_forsam = (255*image_formaskrcnn).type(torch.uint8)[None, :, :, :] # for SAM, input is [B, 3, H, W] (and not (H, W, C) anymore)
    # sam_result = mask_generator.generate(image_forsam.to(torch.device('cpu')), [boxes.to(torch.device('cpu'))]) # TODO: implement for full functionality on GPU. check line 100 in bbox_prompt_mask_generator.py
    sam_result = mask_generator.generate(image_forsam, [boxes])
    # update result with sam masks. since there is only one image here, simply take the first index (sam_result['segmentation'][0])
    masks = torch.as_tensor(sam_result['segmentation'][0][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result from (N, num_classes, H, W) also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension  
    result[0]['masks'] = torch.as_tensor(masks, dtype=torch.float)
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
    boxes = []
    imgs_forsam = []
    mask_idc = []
    for r_idx, res in enumerate(result):
        if len(res['boxes']) == 0: # empty boxes..
            # print('empty box')
            continue
        # boxes.append(res['boxes'].to(torch.device('cpu')))
        # imgs_forsam.append((255*images_formaskrcnn[r_idx]).type(torch.uint8).to(torch.device('cpu')))
        # on the GPU:
        boxes.append(res['boxes'])
        imgs_forsam.append((255*images_formaskrcnn[r_idx]).type(torch.uint8))
        mask_idc.append(r_idx)
    if len(boxes) > 0:
        imgs_forsam = torch.stack(imgs_forsam) # to shape [N, C, H, W]
    else:
        # then, we simply return the result. this happens if we have a full batch of empty images
        return result
    # torch.stack(boxes)
    sam_results = mask_generator.generate(imgs_forsam, boxes) # TODO: implement for full functionality on GPU. check line 100 in bbox_prompt_mask_generator.py
    #### batch processing
    masks = [torch.as_tensor(m[:, 1:].sum(axis=1) > 0)[:,None,:,:] for m in sam_results['segmentation']] # torch.as_tensor(sam_result['segmentation'][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension
    m_idx = 0
    for r_idx, res in enumerate(result):
        if len(res['boxes']) == 0: # empty boxes..
            # print('empty box')
            continue
        # else, we replace mask result with the next mask in the list.       
        res['masks'] = torch.as_tensor(masks[m_idx], dtype=torch.float32)
        # after, we increase the index of the mask. we do this like this to make sure that empty boxes are simply ignored.    
        m_idx = m_idx + 1

    # # prompt sam with mask r-cnn boxes (faster r-cnn)
    # for res_idx, im in enumerate(images_formaskrcnn):
    #     if len(result[res_idx]['boxes']) == 0: # empty boxes..
    #         # print('empty box')
    #         continue
    #     # images_forsam = [(255*im).permute(1,2,0) for im in images_formaskrcnn] # for SAM, input is (H, W, C)
    #     # sam_result = mask_generator.generate(torch.stack(images_forsam), torch.stack(boxes))
    #     image_forsam = (255*im).permute(1,2,0).numpy().astype('uint8') # for SAM, input is (H, W, C)
    #     sam_result = mask_generator.generate(image_forsam, result[res_idx]['boxes'])

    #     # update result with sam masks
    #     masks = torch.as_tensor(sam_result['segmentation'][:, 1:].sum(axis=1) > 0)[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension
    #     result[res_idx]['masks'] = torch.as_tensor(masks, dtype=torch.float32)
    #     # masks.shape: torch.Size([100, 1, 224, 224])
    #     # masks.dtype: torch.float32        
 
    # return the updated result
    return result

def prompt_SamPoints(mask_generator, image_formaskrcnn, rcnnsam_args, cpu_output=False):
    '''
    remember: the returned boxes are not correct! they are simply the whole image: [  0.,   0., 224., 224.] (in XYXY format)
    '''
    # imgs_forsam = []
    # for r_idx in range(len(images_formaskrcnn)):
    #     imgs_forsam.append((255*images_formaskrcnn[r_idx]).type(torch.uint8))
    # imgs_forsam = torch.stack(imgs_forsam) # to shape [N, C, H, W]

    # initialize result:
    # list of one dict
    result = {}

    # works only for single image
    image_forsam = (255*image_formaskrcnn).permute(1,2,0).type(torch.uint8) # for points, we need (W, H, C)

    sam_result = mask_generator.generate(image_forsam.detach().cpu().numpy())
    # sort by score:
    sam_result = sorted(
    sam_result, key=lambda x: x["predicted_iou"], reverse=True
    )   

    if len(sam_result) > 0:
        # output: dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box', 'class_id', 'logits_mask'])
        masks = np.array([mask["segmentation"] for mask in sam_result])[:,None,:,:] # rcnn result['masks'] has shape (N, 1, H, W). we reduce sam result also to (N, 1, H, W) by summing over the class dimension and then adding again a dimension  
        iouscore = np.array([mask["predicted_iou"] for mask in sam_result]) 
        class_id = np.array([mask['class_id'] for mask in sam_result])
        box = np.array([mask['crop_box'] for mask in sam_result])
        result['masks'] = torch.as_tensor(masks, dtype=torch.float32).to(rcnnsam_args['device']) # shape (N, 1, W, H)
        result['scores'] = torch.as_tensor(iouscore, dtype=torch.float32).to(rcnnsam_args['device'])
        result['labels'] = torch.as_tensor(class_id, dtype=torch.int64).to(rcnnsam_args['device'])
        # calculate boxes (by taking the mask and drawing a box around it)
        boxes = []
        for i in range(len(sam_result)):
            pos = np.where(masks[i,0,:,:]) # take the i-th mask
            # it may happen that after augmentation, one mask is empty
            # check this here
            # pos is a tuple with arrays for each dimension
            if len(pos[0]) != 0 or len(pos[1]) != 0:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                # append boxes
                boxes.append([xmin, ymin, xmax, ymax])
        # write to result
        result['boxes'] = torch.as_tensor(boxes, dtype=torch.float32).to(rcnnsam_args['device']) # (N, 4) these are not the correct boxes, but who cares just for precision recall of the masks.
    
    else:
        # construct an empty results dict:
        result = {'boxes': torch.tensor(np.zeros((0, 4))).to(rcnnsam_args['device']),
            'labels': torch.tensor([]).to(rcnnsam_args['device']),
            'scores': torch.tensor([] ).to(rcnnsam_args['device']),
            'masks': torch.tensor(np.zeros((0, 1, rcnnsam_args['img_size'], rcnnsam_args['img_size']))).to(rcnnsam_args['device'])}

    # convert to cpu:
    if cpu_output:
        for key, value in result.items():
            result[key] = value.cpu()

    return result # list of dictionaries with keys dict_keys(['boxes', 'labels', 'scores', 'masks'])




#%%
# main

# from Javier
# dataset_folder = '/Users/javier/Documents/datasets/europa/'

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_forsam = dataset[0]['image'].permute(1,2,0).numpy() # for SAM, input is (H, W, C)
    image_forsam = torch.tensor(image_forsam).to(device)
    image_formaskrcnn = dataset[0]['image']/255
    image_formaskrcnn = image_formaskrcnn.to(device)


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
            'model_name': "./ckpts/Mask_R-CNN_v1_1_17ESREGMAP02_part01_run10_end_model.pt",
            'minsize': 200,
            'maxsize': 300, 
            'device': device,
            'sam_modus': 'vit_b'
        }
    # test first with Point prompting:
    rcnnsam_args['ckpt_path'] = './ckpts/instseg_trainval_object_noign.pt'
    mask_generator_points = get_samPoints(rcnnsam_args)
    result_prompt = prompt_SamPoints(mask_generator_points, image_formaskrcnn, rcnnsam_args)
    # prompt with empty image:
    result_prompt_empty = prompt_SamPoints(mask_generator_points, torch.zeros(image_formaskrcnn.shape).to(rcnnsam_args['device']), rcnnsam_args)
    
    # rcnnSAM
    mask_generator, box_generator = get_rcnnSAM(rcnnsam_args)
    result = get_rcnnsam_output(mask_generator, box_generator, image_formaskrcnn)
    # with empty image:
    resultemp = get_rcnnsam_output(mask_generator, box_generator, torch.zeros(image_formaskrcnn.shape).to(rcnnsam_args['device']))

    # instances = (sam_result['segmentation'][:, 1:].sum(axis=1) > 0) # of shape (N, 224, 224) with N number of masks

    instances = result[0]['masks'][:,0,:,:].cpu().detach().numpy().astype(bool) # for visualization, needs to be boolean
    boxes = result[0]['boxes']

    # The rest is for visualization. Completely optional
    import supervision as sv

    box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.red())
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    instance_detections = sv.Detections(xyxy=boxes.cpu().detach().numpy(), mask=instances, class_id=np.zeros(len(boxes)))
    annotated_image_semantic = mask_annotator.annotate(scene=image_forsam.cpu().numpy().copy(), detections=instance_detections)
    inp = box_annotator.annotate(scene=image_forsam.cpu().numpy().copy(), detections=instance_detections)

    sv.plot_images_grid(
        images=[inp, annotated_image_semantic],
        grid_size=(1, 2),
        titles=['source image', 'instance segmentation'],
    )


    #######################################################
    # test batch processing:
    rcnnsam_args['ckpt_path'] = './ckpts/instseg_bb.pt'
    images_formaskrcnn = [image_formaskrcnn, image_formaskrcnn, torch.zeros(image_formaskrcnn.shape).to(rcnnsam_args['device']), image_formaskrcnn, image_formaskrcnn]
    results = batch_process_rcnnsam(mask_generator, box_generator, images_formaskrcnn)
    # test with a full batch of empty images
    emptim = torch.zeros(image_formaskrcnn.shape).to(rcnnsam_args['device'])
    results = batch_process_rcnnsam(mask_generator, box_generator, [emptim, emptim, emptim, emptim, emptim])

    ###########
    # loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=ListCollate(['bboxes']))

    # data = next(iter(loader))
    # image = data['image'].to(device)  # [B, 3, H, W]
    # boxes = data['bboxes']  # [Bx[N, 4]]
    # boxes = [box.to(device) for box in boxes]
    ###########

    # instances = results[1]['masks'][:,0,:,:].numpy().astype(bool) # for visualization, needs to be boolean
    # boxes = results[1]['boxes']

    # # The rest is for visualization. Completely optional
    # import supervision as sv

    # box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED)
    # mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    # instance_detections = sv.Detections(xyxy=boxes.cpu().detach().numpy(), mask=instances, class_id=np.zeros(len(boxes)))
    # annotated_image_semantic = mask_annotator.annotate(scene=image_forsam.cpu().numpy().copy(), detections=instance_detections)
    # inp = box_annotator.annotate(scene=image_forsam.copy(), detections=instance_detections)

    # sv.plot_images_grid(
    #     images=[inp, annotated_image_semantic],
    #     grid_size=(1, 2),
    #     titles=['source image', 'instance segmentation'],

    # )

    # The result is a dict with the following keys:
    # - 'segmentation': List[torch.Tensor] with the mask
    # - 'bbox': List[torch.Tensor] with the bounding box in xywh format
    # - 'predicted_iou': float with the predicted iou

    # The rest is for visualization. Completely optional
    # import supervision as sv

    # box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED)
    # mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    masks = [result['masks'].cpu().numpy().astype(bool) for result in results] # List[torch.Tensor]
    images = [image_forsam.cpu().detach().numpy(), image_forsam.cpu().detach().numpy(), np.zeros(image_forsam.shape), image_forsam.cpu().detach().numpy(), image_forsam.cpu().detach().numpy()] # image_forsam.cpu().detach().numpy()
    boxes = [result['boxes'].type(torch.int).cpu() for result in results] 
    for i, mask in enumerate(masks):
        image = images[i].copy() #.transpose(1, 2, 0)
        instances = mask[:,0,:,:] # (mask[:, 1:].sum(axis=1) > 0).detach().cpu().numpy()

        instance_detections = sv.Detections(xyxy=boxes[i].cpu().detach().numpy(), mask=instances,
                                            class_id=np.zeros(len(boxes[i])))
        if len(instance_detections.xyxy) == 0:
            print('no detections for image {}'.format(i))
            continue
        annotated_image_semantic = mask_annotator.annotate(scene=image.copy(),
                                                        detections=instance_detections)
        image = box_annotator.annotate(scene=image.copy(), detections=instance_detections)

        sv.plot_images_grid(
            images=[image, annotated_image_semantic],
            grid_size=(1, 2),
            titles=['source image', 'instance segmentation']
        )


# %% from javier:
from segment_anything.bbox_prompt_mask_generator import SamBBoxMaskGenerator
from segment_anything import sam_model_registry
import numpy as np
import torch

from src.datasets import VanillaGalileoDataset
from src.datasets.dataset_utils import ListCollate

def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy

device = "cpu"# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset_folder = '/Users/javier/Documents/datasets/europa/'
dataset_folder = '../datasets_2024-05-31'
dataset = VanillaGalileoDataset(dataset_folder, 'test')
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=ListCollate(['bboxes']))

# IMAGE_NAME = './segment_anything/12ESFRTPLT01EXCERPT1_Equi-cog_15.png'
sam = sam_model_registry['vit_b'](checkpoint='./ckpts/instseg_bb.pt', num_classes=5, image_size=224)
mask_generator = SamBBoxMaskGenerator(sam)

data = next(iter(loader))
image = data['image'].to(device)  # [B, 3, H, W]
boxes = data['bboxes']  # [Bx[N, 4]]
boxes = [box.to(device) for box in boxes]

sam_result = mask_generator.generate(image, boxes)

# The result is a dict with the following keys:
# - 'segmentation': List[torch.Tensor] with the mask
# - 'bbox': List[torch.Tensor] with the bounding box in xywh format
# - 'predicted_iou': float with the predicted iou

# The rest is for visualization. Completely optional
import supervision as sv

box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.RED)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

masks = sam_result['segmentation']  # List[torch.Tensor]
images = image.detach().cpu().numpy()
for i, mask in enumerate(masks):
    image = images[i].copy().transpose(1, 2, 0)
    instances = (mask[:, 1:].sum(axis=1) > 0).detach().cpu().numpy()

    instance_detections = sv.Detections(xyxy=boxes[i].numpy(), mask=instances,
                                        class_id=np.zeros(len(boxes[i])))
    annotated_image_semantic = mask_annotator.annotate(scene=image.copy(),
                                                       detections=instance_detections)
    image = box_annotator.annotate(scene=image.copy(), detections=instance_detections)

    sv.plot_images_grid(
        images=[image, annotated_image_semantic],
        grid_size=(1, 2),
        titles=['source image', 'instance segmentation']
    )

# %%
