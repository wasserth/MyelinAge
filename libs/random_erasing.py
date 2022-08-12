""" Random Erasing (Cutout)

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
"""
import sys
import random
import math
import torch

from monai.transforms.transform import Transform


def _get_pixels(per_pixel, rand_color, patch_size, mean=0, std=1.0,
                dtype=torch.float32, device='cpu'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_(mean, std)
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1, 1), dtype=dtype, device=device).normal_(mean, std)
    else:
        return torch.zeros((patch_size[0], 1, 1, 1), dtype=dtype, device=device)


# Working with monai transforms
class RandomErasing():
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """
    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cpu'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, dtype):
        if random.random() > self.probability:
            return
        if self.rand_color is True or self.per_pixel is True:
            mean, std = img.mean(), img.std()/4
        else: 
            mean, std = 0, 1

        chan, img_x, img_y, img_z = img.size()
        area = img_x * img_y * img_z
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio_xy = math.exp(random.uniform(*self.log_aspect_ratio))
                aspect_ratio_yz = random.uniform(0.5, 1.9)  # higher and lower will lead to more failed attempts

                if img_z == 1:
                    x = round(math.sqrt(target_area * aspect_ratio_xy))
                    y = round(math.sqrt(target_area / aspect_ratio_xy))
                    z = 1
                else:
                    x = round((target_area * aspect_ratio_xy**2 * aspect_ratio_yz)**(1/3))
                    y = round(((target_area * aspect_ratio_yz) / aspect_ratio_xy)**(1/3))
                    z = round(target_area / (x*y))

                if attempt == 9:
                    print("WARNING: Could not find a cutout within 10 attempts")

                if x < img_x and y < img_y and z <= img_z:
                    x_start = random.randint(0, img_x - x)
                    y_start = random.randint(0, img_y - y)
                    z_start = random.randint(0, img_z - z) if z > 1 else 0
                    img[:, x_start:x_start+x, y_start:y_start+y, z_start:z_start+z] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, x, y, z), mean, std,
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.shape) == 3:  # 2d image [c,x,y]
            input = input[:,:,:,None]  # make 3d
            self._erase(input, input.dtype)
            input = input[:,:,:,0]  # back to 2d
        else:  # [c,x,y,z]
            self._erase(input, input.dtype)
        return input


if __name__ == "__main__":
    import nibabel as nib

    file_in = sys.argv[1]
    file_out = sys.argv[2]

    img_in = nib.load(file_in)
    data = img_in.get_fdata()

    # 2d
    data = data[:, :, data.shape[2] // 2]
    data = data[None, :, :]  # add channel dim

    # 3d
    # data = data[None, :, :, :]

    data = torch.from_numpy(data)

    # mode: const or pixel (for noise)
    # max_area: 1/3 a lot of black
    # min_area: little black, but still clearly visible
    # re = RandomErasing(probability=1, mode="const", min_count=1, max_count=5, max_area=1/5, device="cpu")
    re = RandomErasing(probability=1, mode="const", min_count=1, max_count=1, max_area=1/3, device="cpu")
    print("Erasing...")
    data_out = re(data)
    print("Done")

    data_out = data_out.cpu().numpy()

    data_out = data_out[0,:,:]
    nib.save(nib.Nifti1Image(data_out, img_in.affine), file_out)
