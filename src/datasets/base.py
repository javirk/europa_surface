import torch
import random
import numpy as np
from scipy import ndimage
from skimage import measure

from segment_anything.utils.amg import batched_mask_to_box


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

    @staticmethod
    def perturb_bouding_box(bbox: torch.Tensor, original_size: (int, int), max_noise: int = 5) -> torch.Tensor:
        """
        Perturb each side of the bounding box with random noise. Standard deviation equal to 10% of the box sidelength,
        to a maximum of max_noise (5) pixels (in SAM this is done with 20 pixels, but images are 1024px. I transformed
        it to 5 pixels for 224px images)

        Parameters:
        bbox (torch.Tensor): Bounding box in XYXY format (x_min, y_min, x_max, y_max).
        original_size (tuple): Original image size as (width, height).
        max_noise (int): Maximum noise in pixels.

        Returns:
        torch.Tensor: Perturbed bounding box.
        """
        initial_bbox = bbox.clone()
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)
        # Calculate side lengths
        width = bbox[..., 2] - bbox[..., 0]
        height = bbox[..., 3] - bbox[..., 1]

        # Calculate standard deviation for the noise (10% of the side length)
        std_x = 0.1 * width
        std_y = 0.1 * height
        std = torch.stack([std_x, std_y, std_x, std_y], dim=-1)
        # std = torch.repeat_interleave(std, repeats=bbox.shape[0], dim=0)

        # Generate random noise for each coordinate
        noise = torch.normal(mean=0, std=std)

        # Clamp the noise to max_noise
        noise = torch.clamp(noise, -max_noise, max_noise)

        # Apply noise to the bounding box coordinates
        perturbed_bbox = (bbox + noise).to(bbox.dtype)

        # Crop the bounding box to be within image boundaries
        perturbed_bbox[..., 0] = torch.clamp(perturbed_bbox[..., 0], 0, original_size[0])  # x_min
        perturbed_bbox[..., 1] = torch.clamp(perturbed_bbox[..., 1], 0, original_size[1])  # y_min
        perturbed_bbox[..., 2] = torch.clamp(perturbed_bbox[..., 2], 0, original_size[0])  # x_max
        perturbed_bbox[..., 3] = torch.clamp(perturbed_bbox[..., 3], 0, original_size[1])  # y_max

        # Ensure that the perturbed_bbox is still a valid box (x_min < x_max and y_min < y_max)
        perturbed_bbox[..., 2] = torch.max(perturbed_bbox[..., 2], perturbed_bbox[..., 0] + 1)  # Ensure x_max > x_min
        perturbed_bbox[..., 3] = torch.max(perturbed_bbox[..., 3], perturbed_bbox[..., 1] + 1)  # Ensure y_max > y_min

        # Reshape perturbed_bbox to the inital shape
        if initial_bbox.dim() == 1:
            perturbed_bbox = perturbed_bbox.squeeze(0)

        return perturbed_bbox

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
        if self.ignore_outside_pixels:
            mask_ignore_index = torch.ones_like(mask) * self.ignore_index
        else:
            mask_ignore_index = torch.zeros_like(mask)
        val_point = mask[0, point[1], point[0]]
        instance = instance_mask[0, point[1], point[0]]
        mask_ignore_index[instance_mask == instance] = val_point
        mask_instance = mask_ignore_index.clone()
        mask_ignore_index[instance_mask == 0] = 0
        mask_ignore_index = mask_ignore_index.unsqueeze(0)
        return mask_ignore_index, mask_instance

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

        if len(mask.unique()) == 1 or (len(instance_mask.unique()) == 1 and instance_mask.unique()[0] == 0):
            return None

        # Downsample mask
        mask_downsampled = torch.nn.functional.interpolate(mask[None],
                                                           size=(self.downsampling_size, self.downsampling_size),
                                                           mode='nearest')[0, 0].long()

        # Make mask binary
        mask_binary = (mask > 0).float()

        assert len(bounding_boxes) > 0, f'No bounding boxes found in {img_data}'

        # Point stuff:
        point = np.where(mask_binary == 1)
        if len(point[0]) == 0:
            # There's no fluid. Random point and background label
            point = torch.tensor([random.randint(0, mask_binary.shape[-1]),
                                  random.randint(0, mask_binary.shape[-2])])
            mask_point = torch.zeros_like(mask).unsqueeze(0)
            point_label = torch.tensor([0])
            mask_instance_point = torch.zeros_like(mask).unsqueeze(0)
        else:
            # point_idx = np.random.choice(len(point[0]))
            # point = torch.tensor(
            #     [point[2][point_idx], point[1][point_idx]])  # mask is 1xHxW, so we discard the first dim.

            # Choose the instance, then sample a point from it
            instance_num = np.random.choice(instance_mask.unique()[1:])  # Select from any but background
            temp_mask_instance = (instance_mask[0] == instance_num)
            points = np.where(temp_mask_instance)
            point_idx = np.random.choice(len(points[0]))
            point = torch.tensor([points[1][point_idx], points[0][point_idx]])
            mask_point, mask_instance_point = self.isolate_mask_point(point, mask, instance_mask)
            # Transform the point so that it can be used in SAM
            # point = self.sam_transform.apply_coords_torch(point, mask.shape[-2:]).unsqueeze(0)
            point_label = torch.tensor([1])  # Foreground pixel
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        mask_point_downsampled = torch.nn.functional.interpolate(mask_point,
                                                                 size=(self.downsampling_size, self.downsampling_size),
                                                                 mode='nearest')[0, 0].long()

        # Select one bounding box at random. Zero all the values of the mask outside the bounding box
        # bounding_box = bounding_boxes[np.random.choice(len(bounding_boxes))]
        bb_index = self.ignore_index if self.ignore_outside_pixels else 0
        bounding_box = batched_mask_to_box(mask_instance_point != bb_index)
        assert len(bounding_box) == 1, f'More than one bounding box found?'
        bounding_box = self.perturb_bouding_box(bounding_box[0], mask_instance_point.shape[-2:])
        # bounding_box = bounding_box[0]
        mask_bb = self.isolate_mask_bb(bounding_box, mask)
        # Transform the BB so that it can be used in SAM
        # bounding_box = self.sam_transform.apply_boxes_torch(bounding_box, mask.shape[-2:])
        if len(bounding_box.shape) == 1:
            bounding_box = bounding_box.unsqueeze(0)
        mask_bb_downsampled = torch.nn.functional.interpolate(mask_bb[None],
                                                              size=(self.downsampling_size, self.downsampling_size),
                                                              mode='nearest')[0, 0].long()

        name = self._get_name(img_data)

        return_dict = {'image': img, 'mask_bb': mask_bb.to(torch.long), 'mask_point': mask_point[0].long(),
                       'mask_bb_downsampled': mask_bb_downsampled.to(torch.long),
                       'mask_point_downsampled': mask_point_downsampled.to(torch.long),
                       'boxes': bounding_box, 'point': point, 'point_label': point_label, 'name': name,
                       'image_data': img_data, 'original_size': img.shape[-2:], 'mask': mask.long(),
                       'mask_downsampled': mask_downsampled}
        return return_dict


def plot_return(d):
    import torchvision
    import matplotlib.pyplot as plt

    def show_points(coords, ax, marker_size=200):
        ax.scatter(coords[:, 0], coords[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.)

    im = d['image']
    m_box = d['mask_bb']
    m_point = d['mask_point']
    box = d['boxes']
    point = d['point']
    mask = d['mask']

    # Bring back the box to the original size
    box = box.reshape(-1, 2, 2)
    box[..., 0] = box[..., 0] * (im.shape[-1] / 224)
    box[..., 1] = box[..., 1] * (im.shape[-2] / 224)
    box = box.reshape(-1, 4)
    # Bring back the point to the original size
    point[..., 0] = point[..., 0] * (im.shape[-1] / 224)
    point[..., 1] = point[..., 1] * (im.shape[-2] / 224)

    fix, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(torchvision.utils.draw_bounding_boxes(im, box, colors='red').permute(1, 2, 0))
    show_points(point, ax[0, 0])
    ax[0, 1].imshow(m_box[0])
    ax[1, 0].imshow(m_point[0])
    ax[1, 1].imshow(mask[0])

    ax[0, 0].set_title('Original image')
    ax[0, 1].set_title('Mask BB')
    ax[1, 0].set_title('Mask Point')
    ax[1, 1].set_title('Mask')
    plt.show()
