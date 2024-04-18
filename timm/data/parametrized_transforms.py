from torchvision import transforms
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
        return RandomResizedCropAndInterpolationParametrized(transform.size, transform.scale, transform.ratio, 'random')
    elif isinstance(transform, transforms.RandomHorizontalFlip):
        return RandomHorizontalFlipParametrized(transform.p)
    elif isinstance(transform, RandAugment):
        return RandAugmentParametrized([AugmentOpParametrized(op.name, op.prob, op.magnitude, op.hparams) for op in transform.ops], transform.num_layers, transform.choice_weights)
    elif isinstance(transform, ToNumpy):
        return ToNumpyParametrized()

    assert False, 'Unknown transform {}'.format(type(transform))



class ComposeParametrized:
    """Composes several transforms together. """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToNumpyParametrized:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img


class RandomResizedCropAndInterpolationParametrized(RandomResizedCropAndInterpolation):
    """ Crop the given PIL Image to random size and aspect ratio with random interpolation. """

    def __init__(self, size, scale, ratio, interpolation):
        super().__init__(size, scale, ratio, interpolation)
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

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

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"



class RandAugmentParametrized:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops,
            self.num_layers,
            replace=self.choice_weights is None,
            p=self.choice_weights,
        )
        for op in ops:
            img = op(img)
        return img

    def __repr__(self):
        fs = self.__class__.__name__ + f'(n={self.num_layers}, ops='
        for op in self.ops:
            fs += f'\n\t{op}'
        fs += ')'
        return fs


class AugmentOpParametrized:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
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

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
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
        return self.aug_fn(img, *level_args, **self.kwargs)

    def __repr__(self):
        fs = self.__class__.__name__ + f'(name={self.name}, p={self.prob}'
        fs += f', m={self.magnitude}, mstd={self.magnitude_std}'
        if self.magnitude_max is not None:
            fs += f', mmax={self.magnitude_max}'
        fs += ')'
        return fs