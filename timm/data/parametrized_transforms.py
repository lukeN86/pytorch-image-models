from torchvision import transforms

from timm.data.random_erasing import RandomErasing
from timm.data.transforms import RandomResizedCropAndInterpolation, ToNumpy, interp_mode_to_str
from timm.data.auto_augment import  RandAugment, _HPARAMS_DEFAULT, NAME_TO_OP, LEVEL_TO_ARG, _FILL, _RANDOM_INTERPOLATION, _LEVEL_DENOM
import random
import torch
import numpy as np
from torchvision.transforms import functional as F

def convert_to_parametrized_transform(transform):

    if isinstance(transform, transforms.Compose):
        return ComposeParametrized([convert_to_parametrized_transform(t) for t in transform.transforms])
    elif isinstance(transform, RandomResizedCropAndInterpolation):
        return RandomResizedCropAndInterpolationParametrized(transform.size, transform.scale, transform.ratio, _RANDOM_INTERPOLATION)
    elif isinstance(transform, transforms.RandomHorizontalFlip):
        return RandomHorizontalFlipParametrized(transform.p)
    elif isinstance(transform, RandAugment):
        return RandAugmentParametrized([AugmentOpParametrized(op.name, op.prob, op.magnitude, op.hparams) for op in transform.ops], transform.num_layers, transform.choice_weights)
    elif isinstance(transform, ToNumpy):
        return ToNumpyParametrized()
    elif isinstance(transform, transforms.ToTensor):
        return ToTensorParametrized()
    elif isinstance(transform, transforms.Normalize):
        return NormalizeParametrized(transform.mean, transform.std, transform.inplace)
    elif isinstance(transform, RandomErasing):
        return RandomErasingParametrized(transform.probability, transform.min_area, transform.max_area, transform.min_aspect, transform.max_aspect, transform.mode, transform.min_count, transform.max_count, transform.num_splits, transform.device)

    assert False, 'Unknown transform {}'.format(type(transform))



class ComposeParametrized:
    """Composes several transforms together. """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, grid):
        for t in self.transforms:
            img, grid = t(img, grid)
        return img, grid

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToNumpyParametrized:

    def __call__(self, pil_img, grid):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img, grid


class ToTensorParametrized:
    """Convert a PIL Image or ndarray to tensor and scale the values accordingly. """

    def __call__(self, pic, grid):
        return F.to_tensor(pic), grid

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NormalizeParametrized(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.  """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor, grid):
        return F.normalize(tensor, self.mean, self.std, self.inplace), grid

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"



class RandomErasingParametrized(RandomErasing):
    """ Randomly selects a rectangle region in an image and erases its pixels.  """

    def __init__(
            self,
            probability=0.5,
            min_area=0.02,
            max_area=1/3,
            min_aspect=0.3,
            max_aspect=None,
            mode='const',
            min_count=1,
            max_count=None,
            num_splits=0,
            device='cuda',
    ):
        super().__init__(probability, min_area, max_area, min_aspect, max_aspect, mode, min_count, max_count, num_splits, device)

    def __call__(self, input, grid):
        return super().__call__(input), grid

    def __repr__(self):
        # NOTE simplified state for repr
        fs = self.__class__.__name__ + f'(p={self.probability}, mode={self.mode}'
        fs += f', count=({self.min_count}, {self.max_count}))'
        return fs


class RandomResizedCropAndInterpolationParametrized:
    """ Crop the given PIL Image to random size and aspect ratio with random interpolation. """

    def __init__(self, size, scale, ratio, interpolation):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, img, grid):

        i, j, h, w = RandomResizedCropAndInterpolation.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        cropped_img = F.resized_crop(img, i, j, h, w, self.size, interpolation)
        cropped_grid = F.resized_crop(grid, i, j, h, w, self.size, interpolation, antialias=True)
        return cropped_img, cropped_grid



    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([interp_mode_to_str(x) for x in self.interpolation])
        else:
            interpolate_str = interp_mode_to_str(self.interpolation)
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomHorizontalFlipParametrized(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability. """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, grid):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(grid)

        return img, grid

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"



class RandAugmentParametrized:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img, grid):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )

        # The RandAugment operations only work with PIL images, have to convert the tensor to an array of float32 single channel images
        grid_pil_array = [F.to_pil_image(grid[i, 0, :, :].numpy().astype(np.float32), mode='F') for i in range(grid.size(0))]

        for op in ops:
            img, grid_pil_array = op(img, grid_pil_array)

        grid = torch.stack([F.to_tensor(g) for g in grid_pil_array], dim=0)

        return img, grid

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


# List of operations which modify coordinates
COORDINATE_OPS = [
    'Rotate',
    'ShearX',
    'ShearY',
    'TranslateX',
    'TranslateY',
    'TranslateXRel',
    'TranslateYRel',
]


class AugmentOpParametrized:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.grid_aug_fn = NAME_TO_OP[name] if name in COORDINATE_OPS else None
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution
        self.magnitude_std = self.hparams.get('magnitude_std', 0)
        self.magnitude_max = self.hparams.get('magnitude_max', None)

    def __call__(self, img, grid_pil_array):
        if self.prob < 1.0 and random.random() > self.prob:
            return img, grid_pil_array
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            # magnitude randomization enabled
            if self.magnitude_std == float('inf'):
                # inf == uniform sampling
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        # default upper_bound for the timm RA impl is _LEVEL_DENOM (10)
        # setting magnitude_max overrides this to allow M > 10 (behaviour closer to Google TF RA impl)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0., min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()

        if self.grid_aug_fn is not None:
            # Change shape of the image and the coordinates grid (e.g. TranslateX)
            return self.aug_fn(img, *level_args, **self.kwargs), [self.grid_aug_fn(g, *level_args) for g in grid_pil_array]
        else:
            # This operation does not change shape of the image (e.g. AutoContrast)
            return self.aug_fn(img, *level_args, **self.kwargs), grid_pil_array


    def __repr__(self):
        fs = self.__class__.__name__ + f'(name={self.name}, p={self.prob}'
        fs += f', m={self.magnitude}, mstd={self.magnitude_std}'
        if self.magnitude_max is not None:
            fs += f', mmax={self.magnitude_max}'
        fs += ')'
        return fs