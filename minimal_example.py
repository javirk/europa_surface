import cv2
import torch
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry
import numpy as np

def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]
    return xyxy

IMAGE_NAME = './segment_anything/12ESFRTPLT01EXCERPT1_Equi-cog_15.png'
sam = sam_model_registry['vit_b'](checkpoint='./ckpts/instseg_trainval_object_noign.pt', num_classes=5, image_size=224)
mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.25, min_mask_region_area=200, points_per_side=16)

# Reading with cv2, but you can use whatever you prefer. The image must be in RGB format, uint8 and HWC
image_bgr = cv2.imread(IMAGE_NAME)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # Because cv2 reads as bgr.
input = torch.as_tensor(image_rgb)

sam_result = mask_generator.generate(image_rgb)

# The result is already in sam_result, which is a list of dicts with the following keys:
# - 'segmentation': np.ndarray with the mask
# - 'class_id': int with the class id
# - 'logits_mask': np.ndarray with the logits mask
# - 'bbox': np.ndarray with the bounding box in xywh format
# - 'predicted_iou': float with the predicted iou

# The rest is for visualization. Completely optional
import supervision as sv

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
detections = sv.Detections.from_sam(sam_result=sam_result)  # This does not use the class id
annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections) # Instance segmentation

# Add another image with the semantic segmentation
sorted_generated_masks = sorted(
    sam_result, key=lambda x: x["area"], reverse=True
)

mask = np.array([mask["segmentation"] for mask in sorted_generated_masks])
class_id = np.array([mask['class_id'] for mask in sorted_generated_masks])
sem_mask = np.zeros([5, *mask.shape[1:]])

# Put all the annotations together for semantic segmentation
for i in range(len(mask)):
    sem_mask[class_id[i], mask[i] == 1] = 1

xyxy = np.zeros((sem_mask.shape[0], 4))
semantic_detections = sv.Detections(xyxy=xyxy, mask=sem_mask.astype(bool))

annotated_image_semantic = mask_annotator.annotate(scene=image_bgr.copy(), detections=semantic_detections) # Semantic seg

sv.plot_images_grid(
    images=[image_bgr, annotated_image, annotated_image_semantic],
    grid_size=(1, 3),
    titles=['source image', 'segmented image', 'class segmentation image']
)

