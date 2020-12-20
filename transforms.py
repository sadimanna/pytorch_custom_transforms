import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from PIL import Image
from torch import Tensor
#import torch.functional as F
import torchvision.transforms.functional as F

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

class ColorJitter(nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_transform(self):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms_list = []

        if self.brightness is not None:
            brightness_factor = random.uniform(max(0,1-self.brightness), 1+self.brightness)
            transforms_list.append(transforms.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if self.contrast is not None:
            contrast_factor = random.uniform(max(0,1-self.contrast), 1+self.contrast)
            transforms_list.append((transforms.Lambda(lambda img: F.adjust_contrast(img, contrast_factor))))

        if self.saturation is not None:
            saturation_factor = random.uniform(max(0,1-self.saturation), 1+self.saturation)
            transforms_list.append((transforms.Lambda(lambda img: F.adjust_saturation(img, saturation_factor))))

        if self.hue is not None:
            hue_factor = random.uniform(-1*self.hue, self.hue)
            transforms_list.append((transforms.Lambda(lambda img: F.adjust_hue(img, hue_factor))))

        #random.shuffle(transforms_list)
        transform = transforms.Compose(transforms_list)

        return transform

class HorizontalFlip(nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        super().__init__()
        #self.p = p

    def get_transform(self):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        return transforms.Lambda(lambda img: F.hflip(img))

class RandomRotation(nn.Module):
    """Rotate the image by angle.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample (int, optional): An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
            If input is Tensor, only ``PIL.Image.NEAREST`` and ``PIL.Image.BILINEAR`` are supported.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (list or tuple, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for Pillow>=5.2.0.
            This option is not supported for Tensor input. Fill value for the area outside the transform in the output
            image is always 0.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = resample
        self.expand = expand
        self.fill = fill

    def get_transform(self):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        return transforms.Lambda(lambda img: F.rotate(img, angle, self.resample, self.expand, self.center, self.fill))


class Grayscale(nn.Module):
    """Randomly convert image to grayscale with a probability of p (default 0.1).
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self):
        super().__init__()
        #self.p = p

    def get_transform(self):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """
        #num_output_channels = F._get_image_num_channels(img)
        return transforms.Lambda(lambda img: F.rgb_to_grayscale(img, num_output_channels=F._get_image_num_channels(img)))
        

class GaussianBlur(nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma


    def get_transform(self):
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()

        return transforms.Lambda(lambda img: F.gaussian_blur(img, self.kernel_size, [sigma, sigma]))



class RandomCrop(nn.Module):
    """Crop the given image at a random location.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            Mode symmetric is not yet supported for Tensor inputs.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode


    def get_params(self, img):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F._get_image_size(img)
        #print(h,w)
        th, tw = self.size
        #print(h,w,th,tw)

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            if isinstance(self.padding, Tuple):
                w += 2*self.padding[0]
                h += 2*self.padding[1]
            else:
                w += 2*self.padding
                h += 2*self.padding
        
        # pad the width if needed
        if self.pad_if_needed and w < tw:
            padding = [tw - w, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            w = self.size[1]
        # pad the height if needed
        if self.pad_if_needed and h < th:
            padding = [0, th - h]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            h = self.size[0]

        if w == tw and h == th:
            return img, 0, 0, h, w

        #TOP
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        #LEFT
        j = torch.randint(0, w - tw + 1, size=(1, )).item()

        return img, i, j, th, tw

    def crop(self, params):
        img = params[0]
        params = params[1:]
        old_params = params
        #w, h = F._get_image_size(img)
        #th, tw = self.size
        #print(params)
        if isinstance(img,int):
            print("img type int")
            print(img, old_params)
        return img.narrow(-2,params[0],params[2]).narrow(-1,params[1],params[3])

    def get_transform(self):
        return transforms.Lambda(lambda img: self.crop(self.get_params(img)))
    
