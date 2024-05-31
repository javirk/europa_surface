from segment_anything.bbox_prompt_mask_generator import SamBBoxMaskGenerator
from segment_anything import sam_model_registry
import numpy as np

from src.datasets import VanillaGalileoDataset

def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy


dataset_folder = '/Users/javier/Documents/datasets/europa/'
dataset = VanillaGalileoDataset(dataset_folder, 'train')

# IMAGE_NAME = './segment_anything/12ESFRTPLT01EXCERPT1_Equi-cog_15.png'
sam = sam_model_registry['vit_b'](checkpoint='./ckpts/instseg_bb.pt', num_classes=5, image_size=224)
mask_generator = SamBBoxMaskGenerator(sam)

image = dataset[0]['image'].permute(1,2,0).numpy()
boxes = dataset[0]['bboxes']  # [N, 4]

sam_result = mask_generator.generate(image, boxes)

# The result is a dict with the following keys:
# - 'segmentation': np.ndarray with the mask
# - 'bbox': np.ndarray with the bounding box in xywh format
# - 'predicted_iou': float with the predicted iou

instances = (sam_result['segmentation'][:, 1:].sum(axis=1) > 0)

# The rest is for visualization. Completely optional
import supervision as sv

box_annotator = sv.BoundingBoxAnnotator(color=sv.Color.red())
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

instance_detections = sv.Detections(xyxy=boxes.numpy(), mask=instances, class_id=np.zeros(len(boxes)))
annotated_image_semantic = mask_annotator.annotate(scene=image.copy(), detections=instance_detections)
inp = box_annotator.annotate(scene=image.copy(), detections=instance_detections)

sv.plot_images_grid(
    images=[inp, annotated_image_semantic],
    grid_size=(1, 2),
    titles=['source image', 'instance segmentation'],
)