import sys

import numpy as np
import nibabel as nib
import nibabel.processing
from scipy.ndimage import map_coordinates


def transform_image(target_img_path, moving_img_path, img_out_path, dtype=np.float32):
    """
    Resamples moving_img to the grid of target_img (like 'mrtransform -template x' but results in 
    same affine. mrtransform can have flipped sign in affine.).

    Using dipy.

    Another easier way: 
    import nibabel.processing
    moving_transformed = nibabel.processing.resample_from_to(moving, target)
    """
    from dipy.align.imaffine import AffineMap

    target_img = nib.load(target_img_path)
    moving_img = nib.load(moving_img_path)

    moving = moving_img.get_fdata()
    affine_map = AffineMap(np.eye(4),
                           target_img.get_fdata().shape, target_img.affine,
                           moving.shape, moving_img.affine
                           )
    moving = affine_map.transform(moving, interp='linear')  # only nearest or linear possible
    nib.save(nib.Nifti1Image(moving.astype(dtype), target_img.affine), img_out_path)


def resample_to_new_grid(src: nib.Nifti1Image,
                         target: nib.Nifti1Image,
                         order: int = 3,
                         z_offset: float = 0.0,
                         round_small_vals_to_0: bool = True) -> nib.Nifti1Image:
    """
    Resample the src image to the grid of the target image. 
    Expects 3d images.

    Using my own implementation. 

    order: order of spline interpolation
    z_offset: offset added to the z-coordinate when looking up intensities in src image
    """
    # get coordinates of the target grid
    shape_as_range = tuple([np.arange(i) for i in target.shape])
    coords = np.array(np.meshgrid(*shape_as_range, indexing='ij')).astype(float)
    # Add additional dimension with 1 to each coordinate to make work with 4x4 affine matrix
    addon = np.ones([1, coords.shape[1], coords.shape[2], coords.shape[3]])
    coords = np.concatenate([coords, addon], axis=0)  # shape: [4, x, y, z]
    coords = coords.reshape([4, -1])  # shape: [4, x*y*z]

    # build affine which maps from target grid to source grid
    aff_transf = np.linalg.inv(src.affine) @ target.affine

    # transform the coords from target grid to the space of source image
    coords_src = aff_transf @ coords  # shape: [4, x*y*z]

    # add a custom offset to the z-coordinate. Can be used for multislice dicom images (images which have 
    # a gap between the slices)
    coords_src[2, :] += z_offset

    # reshape to original spatial dimensions
    coords_src = coords_src.reshape((4,) + target.shape)[:3,...]  # shape: [3, x, y, z]

    # Will create a image with the spatial size of coords_src (which is target.shape). Each
    # coordinate contains a place in the source image from which the intensity is taken 
    # and filled into the new image. If the coordinate is not within the range of the source image then
    # will be filled with 0.
    src_transf_data = map_coordinates(src.get_fdata(), coords_src, order=order)

    # remove small negative values introduced by resampling
    if round_small_vals_to_0:
        src_transf_data[(src_transf_data > -0.01) & (src_transf_data < 0.01)] = 0

    return nib.Nifti1Image(src_transf_data, target.affine)


def rigid_registration_dipy(src_img: nib.Nifti1Image, target_img: nib.Nifti1Image):
    """
    Rigidly register src_img to target_img. (using dipy)
    """
    from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                    RigidTransform3D,
                                    AffineTransform3D)

    src = src_img.get_fdata()
    target = target_img.get_fdata()

    if ((src != 0).sum() == 0) or ((target != 0).sum() == 0):
        print("WARNING: src or target image are all 0. Can not do registration.")
        return nib.Nifti1Image(np.zeros(target_img.shape), target_img.affine)

    # First rough alignment -> will speedup rigid registration
    c_of_mass = transform_centers_of_mass(target, target_img.affine,
                                          src, src_img.affine)
    # transformed = c_of_mass.transform(src)

    # sampling_proportion=0.5 (instead of 1.0): speedup 2x
    # nbins=16 (instead of 32): no speedup
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.2)

    # Reducing nr of levels: no speedup
    # Reducing nr of iterations per level: speedup
    # level_iters = [10000, 1000, 100]  # 50s
    level_iters = [100, 50, 5]  # 20s
    sigmas = [3.0, 1.0, 0.0]
    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors, 
                                verbosity=0)

    # starting_affine = c_of_mass.affine
    # translation = affreg.optimize(target, src,
    #                               TranslationTransform3D(), None,
    #                               target_img.affine, src_img.affine,
    #                               starting_affine=starting_affine)
    # transformed = translation.transform(src)

    # starting_affine = translation.affine
    starting_affine = c_of_mass.affine
    rigid = affreg.optimize(target, src,
                            RigidTransform3D(), None,
                            target_img.affine, src_img.affine,
                            starting_affine=starting_affine)
    transformed = rigid.transform(src)

    # starting_affine = rigid.affine
    # affine = affreg.optimize(target, src,
    #                          AffineTransform3D(), None,
    #                          target_img.affine, src_img.affine,
    #                          starting_affine=starting_affine)
    # transformed = affine.transform(src)

    return nib.Nifti1Image(transformed, target_img.affine)


def as_closest_canonical_nifti(path_in, path_out):
    """
    Convert the given nifti file to the closest canonical nifti file.
    """
    img_in = nib.load(path_in)
    img_out = nib.as_closest_canonical(img_in)
    nib.save(img_out, path_out)
    

def undo_canonical(img_can, img_orig):
    """
    Inverts nib.to_closest_canonical()

    img_can: the image we want to move back
    img_orig: the original image because transforming to canonical

    returns image in original space

    https://github.com/nipy/nibabel/issues/1063
    """
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")

    to_canonical = img_ornt  # Same as ornt_transform(img_ornt, ras_ornt)
    from_canonical = ornt_transform(ras_ornt, img_ornt)

    # Same as as_closest_canonical
    # img_canonical = img_orig.as_reoriented(to_canonical)

    return img_can.as_reoriented(from_canonical)


def undo_canonical_nifti(path_in_can, path_in_orig, path_out):
    """
    path_in_can: path to image in canonical space
    path_in_orig: path to image in original space
    path_out: path to output image in original space
    """
    img_can = nib.load(path_in_can)
    img_orig = nib.load(path_in_orig)
    img_out = undo_canonical(img_can, img_orig)
    nib.save(img_out, path_out)


if __name__ == "__main__":
    import nibabel.processing
    moving = nib.load(sys.argv[1])
    target = nib.load(sys.argv[2])
    out = sys.argv[3]
    moving_transformed = nibabel.processing.resample_from_to(moving, target)
    nib.save(moving_transformed, out)