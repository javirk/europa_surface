import torch
import random
import numpy as np
from scipy import ndimage
from skimage import measure


class DatasetBase(torch.utils.data.Dataset):

    def get_bounding_boxes(self, mask):
        if mask.sum() == 0:
            # There's no fluid. Return a random bounding box
            min_x = random.randint(0, int(mask.shape[-1] * 0.8))  # 0.8 to avoid too small boxes at the edges
            min_y = random.randint(0, int(mask.shape[-2] * 0.8))
            max_x = random.randint(min_x, mask.shape[-1])
            max_y = random.randint(min_y, mask.shape[-2])
            return torch.tensor([[min_x, min_y, max_x, max_y]]), mask

        # Label connected components
        labeled_mask = measure.label(mask, connectivity=2)

        # Initialize an empty list to store bounding boxes
        bounding_boxes = []

        if random.random() < 0.3:
            # Bounding box corresponds to pockets of fluid
            num_labels = len(np.unique(labeled_mask))
            # Loop through each labeled region
            for label in range(1, num_labels):  # Start from 1 to exclude background (0)
                # Calculate bounding box coordinates for each blob within the class
                for blob_coordinates in ndimage.find_objects(labeled_mask == label):
                    min_y, max_y = blob_coordinates[0].start, blob_coordinates[0].stop
                    min_x, max_x = blob_coordinates[1].start, blob_coordinates[1].stop
                    bbox = torch.tensor([min_x, min_y, max_x, max_y])
                    bbox = self.perturbate_bbox(bbox, labeled_mask.shape[-2:])
                    bounding_boxes.append(bbox)
        else:
            # Bounding box corresponds to the whole mask
            blob_coordinates = ndimage.find_objects((labeled_mask != 0).astype(int))[0]
            min_y, max_y = blob_coordinates[0].start, blob_coordinates[0].stop
            min_x, max_x = blob_coordinates[1].start, blob_coordinates[1].stop
            bbox = torch.tensor([min_x, min_y, max_x, max_y])
            bbox = self.perturbate_bbox(bbox, labeled_mask.shape[-2:])
            bounding_boxes.append(bbox)

        bounding_boxes = torch.stack(bounding_boxes, dim=0)
        return bounding_boxes, labeled_mask

    def perturbate_bbox(self, bbox: torch.Tensor, original_size: (int, int)) -> torch.Tensor:
        bbox[0] = max(0, bbox[0] - random.randint(0, self.bbox_shift))  # xmin
        bbox[1] = max(0, bbox[1] - random.randint(0, self.bbox_shift))  # ymin
        bbox[2] = min(original_size[1], bbox[2] + random.randint(0, self.bbox_shift))  # xmax
        bbox[3] = min(original_size[0], bbox[3] + random.randint(0, self.bbox_shift))  # ymax
        return bbox

    def isolate_mask_bb(self, bounding_box, mask):
        min_x, min_y, max_x, max_y = bounding_box
        # Return the same mask but with ignore_index outside the bounding box
        mask_zero = torch.zeros_like(mask)
        mask_zero[:, min_y:max_y, min_x:max_x] = 1
        mask = mask * mask_zero
        # Set values outside the bb to ignore_index
        # Create a mask for values outside the bounding box
        outside_bbox_mask = torch.zeros_like(mask, dtype=torch.bool)
        outside_bbox_mask[:, :min_y, :] = True
        outside_bbox_mask[:, :, :min_x] = True
        outside_bbox_mask[:, max_y:, :] = True
        outside_bbox_mask[:, :, max_x:] = True
        mask[outside_bbox_mask] = self.ignore_index
        return mask

    def isolate_mask_point(self, point, mask, instance_mask):
        if type(mask) == np.ndarray:
            mask = torch.from_numpy(mask)
        mask_zero = torch.zeros_like(mask)
        val_point = mask[0, point[1], point[0]]
        instance = instance_mask[0, point[1], point[0]]
        mask_zero[instance_mask == instance] = val_point
        mask_zero = mask_zero.unsqueeze(0)
        return mask_zero

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return self.__class__.__name__

    def __getitem__(self, idx):
        img_data = self.imgs[idx]  # path, slice number
        img, mask, instance_mask, bounding_boxes = self._read_image(img_data)

        img = torch.repeat_interleave(img, repeats=3, dim=0)
        # img = img / 255.

        if self.transforms is not None:
            img, mask, instance_mask, bounding_boxes = self.transforms(img, mask, instance_mask, bounding_boxes)

        # Downsample mask
        mask_downsampled = torch.nn.functional.interpolate(mask[None],
                                                           size=(self.downsampling_size, self.downsampling_size),
                                                           mode='nearest')[0, 0].long()

        # Make mask binary
        mask_binary = (mask > 0).float()

        assert len(bounding_boxes) > 0, f'No bounding boxes found in {img_data}'

        # Select one bounding box at random. Zero all the values of the mask outside the bounding box
        bounding_box = bounding_boxes[np.random.choice(len(bounding_boxes))]
        mask_bb = self.isolate_mask_bb(bounding_box, mask)
        # Transform the BB so that it can be used in SAM
        # bounding_box = self.sam_transform.apply_boxes_torch(bounding_box, mask.shape[-2:])
        if len(bounding_box.shape) == 1:
            bounding_box = bounding_box.unsqueeze(0)
        mask_bb_downsampled = torch.nn.functional.interpolate(mask_bb[None],
                                                              size=(self.downsampling_size, self.downsampling_size),
                                                              mode='nearest')[0, 0].long()

        # Point stuff:
        point = np.where(mask_binary == 1)
        if len(point[0]) == 0:
            # There's no fluid. Random point and background label
            point = torch.tensor([random.randint(0, mask_binary.shape[-1]),
                                  random.randint(0, mask_binary.shape[-2])])
            mask_point = torch.zeros_like(mask).unsqueeze(0)
            point_label = torch.tensor([0])
        else:
            point_idx = np.random.choice(len(point[0]))
            point = torch.tensor(
                [point[2][point_idx], point[1][point_idx]])  # mask is 1xHxW, so we discard the first dim.
            mask_point = self.isolate_mask_point(point, mask, instance_mask)
            # Transform the point so that it can be used in SAM
            # point = self.sam_transform.apply_coords_torch(point, mask.shape[-2:]).unsqueeze(0)
            point_label = torch.tensor([1])  # Foreground pixel
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        mask_point_downsampled = torch.nn.functional.interpolate(mask_point,
                                                                 size=(self.downsampling_size, self.downsampling_size),
                                                                 mode='nearest')[0, 0].long()

        name = self._get_name(img_data)

        return {'image': img, 'mask_bb': mask_bb.to(torch.long), 'mask_point': mask_point[0].long(),
                'mask_bb_downsampled': mask_bb_downsampled.to(torch.long),
                'mask_point_downsampled': mask_point_downsampled.to(torch.long),
                'boxes': bounding_box, 'point': point, 'point_label': point_label, 'name': name,
                'image_data': img_data, 'original_size': img.shape[-2:], 'mask': mask.long(),
                'mask_downsampled': mask_downsampled}
