import torch
import torch.utils.data
import torch.nn.functional as F


def none_collate(batch):
    """
    Puts each data field into a tensor with outer dimension batch size
    :param batch: list[Dict()]
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    batch = [{key: value for key, value in b.items() if value is not None} for b in batch]
    return torch.utils.data.dataloader.default_collate(batch)


def custom_collate(batch):
    tensors, masks, dictionaries = zip(*batch)
    # Stack tensors into a batched tensor
    batched_tensor = torch.stack(tensors, dim=0)
    batched_masks = torch.stack(masks, dim=0)
    # Return a tuple of batched tensor and a list of dictionaries
    return batched_tensor, batched_masks, list(dictionaries)


def random_class_permutation(mask, num_classes, strength=0.05):
    assert 0 <= strength <= 1, 'Strength should be between 0 and 1'
    batch_size, height, width = mask.size(0), mask.size(1), mask.size(2)
    device = mask.device

    # Flatten the mask tensor
    flattened_mask = mask.view(batch_size, -1)

    # Calculate the number of pixels to permute
    num_pixels_to_permute = int(strength * flattened_mask.size(1))

    # Generate random indices for selecting pixels to permute
    permute_indices = torch.randint(0, flattened_mask.size(1), (batch_size, num_pixels_to_permute), device=device)

    # Generate random class permutation
    random_classes = torch.randint(0, num_classes, (batch_size, num_pixels_to_permute), dtype=mask.dtype, device=device)

    # Update the flattened mask with random class permutation
    flattened_mask.scatter_(1, permute_indices, random_classes)

    # Reshape the updated mask tensor to its original shape
    updated_mask = flattened_mask.view(batch_size, height, width)

    return updated_mask


def random_mask_shift(mask, max_shift=10):
    batch_size, height, width = mask.size(0), mask.size(1), mask.size(2)

    # Generate random shifts for height and width
    shift_height = torch.randint(-max_shift, max_shift + 1, (batch_size,))
    shift_width = torch.randint(-max_shift, max_shift + 1, (batch_size,))

    # Clamp shifts to ensure they are within the valid range
    shift_height = torch.clamp(shift_height, -height + 1, height - 1)
    shift_width = torch.clamp(shift_width, -width + 1, width - 1)

    # Shift the mask tensor
    shifted_mask = mask.clone()
    for i in range(batch_size):
        shifted_mask[i] = torch.roll(shifted_mask[i], shifts=int(shift_height[i]), dims=0)
        shifted_mask[i] = torch.roll(shifted_mask[i], shifts=int(shift_width[i]), dims=1)

    return shifted_mask


def perturb_mask(mask, num_classes, max_shift=10, permutation_strength=0.01):
    """
    Apply erosion, dilation, pixel shifting and class swapping
    :param mask: Shape: (batch_size, height, width)
    :param num_classes: Number of classes in the mask
    :param max_shift: Maximum shift for pixel shifting
    :param permutation_strength: Strength of class swapping
    :return: perturbed mask. Shape: (batch_size, height, width)
    """
    batch_size, height, width = mask.shape
    mask_dtype = mask.dtype
    # Apply erosion
    mask = mask.float()
    mask = mask.unsqueeze(1)
    mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    mask = torch.nn.functional.interpolate(mask, size=(height, width), mode='nearest')
    mask = mask.squeeze(1)

    # Apply shifting
    mask = random_mask_shift(mask, max_shift=max_shift)

    # Apply class swapping: get a random number of pixels and swap their value
    mask = random_class_permutation(mask, num_classes, strength=permutation_strength)

    mask = mask.to(mask_dtype)
    return mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mask = torch.zeros((1, 512, 512), dtype=torch.long)
    mask[0, 100:200, 100:200] = 1
    mask[0, 300:400, 300:400] = 2
    perturbed_mask = perturb_mask(mask, 3)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(mask.squeeze(0))
    ax1.set_title('Original Mask')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(perturbed_mask.squeeze(0))
    ax2.set_title('Perturbed Mask')
    plt.show()
