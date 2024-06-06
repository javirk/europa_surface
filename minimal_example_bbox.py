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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_folder = '/Users/javier/Documents/datasets/europa/'
dataset = VanillaGalileoDataset(dataset_folder, 'train')
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
