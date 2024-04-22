import numbers
from collections.abc import Sequence

import torch
from torch import Tensor
from torchvision.transforms._functional_tensor import _cast_squeeze_in, _cast_squeeze_out, _assert_image_tensor


def gaussian_noise(img: Tensor, mean: float, sigma: float) -> Tensor:
    if not (isinstance(img, torch.Tensor)):
        raise TypeError(f"img should be Tensor. Got {type(img)}")

    _assert_image_tensor(img)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [dtype])
    # add the gaussian noise with the given mean and sigma.
    noise = sigma * torch.randn_like(img) + mean
    img = img + noise

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian noise to the image with specified mean and standard deviation.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        mean (float or sequence): Mean of the sampling gaussian distribution .
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            sampling the gaussian noise. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.
    Returns:
        PIL Image or Tensor: Input image perturbed with Gaussian Noise.
    """

    def __init__(self, mean, sigma=(0.1, 0.5)):
        super().__init__()

        if mean < 0:
            raise ValueError("Mean should be a positive number")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.mean = mean
        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            image (PIL Image or Tensor): image to be perturbed with gaussian noise.
            mask (PIL Image or Tensor): mask.
        Returns:
            PIL Image or Tensor: Image added with gaussian noise.
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        output = gaussian_noise(image, self.mean, sigma)
        return output, mask

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma})"
        return s
