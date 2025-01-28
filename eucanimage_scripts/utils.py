import os
import pydicom
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform

def get_affine_matrix_from_dicom(dcm_series_path):

    # Get the DICOM series as a 3D volume
    dicom_files = sorted(os.listdir(dcm_series_path))
    dicom_images = [pydicom.dcmread(os.path.join(dcm_series_path, f)) for f in dicom_files]

    # Extract DICOM orientation, spacing, and origin
    first_dicom = dicom_images[0]
    dicom_orientation = np.array(first_dicom.ImageOrientationPatient).reshape(2, 3)  # (2, 3)
    dicom_spacing = np.array(list(first_dicom.PixelSpacing) + [first_dicom.SliceThickness])  # (3,)
    dicom_origin = np.array(first_dicom.ImagePositionPatient)  # (3,)
    
    # Ensure dicom_orientation is a full 3x3 matrix
    dicom_orientation = np.vstack([dicom_orientation, np.cross(dicom_orientation[0], dicom_orientation[1])])

    # Build the DICOM affine matrix
    dicom_affine = np.eye(4)
    dicom_affine[:3, :3] = dicom_orientation * dicom_spacing[:, None]  # Scale orientation by spacing
    dicom_affine[:3, 3] = dicom_origin

    # Convert from DICOM LPI to NIfTI RPS
    # Flip the x-axis (L->R) and z-axis (S->P), keeping the y-axis unchanged
    lpi_to_rps = np.diag([-1, 1, 1, 1])  # Flip x (L->R), keep y (P->P), flip z (S->P)
    rps_affine = lpi_to_rps @ dicom_affine

    return rps_affine

def reorient_and_resample_mask(nifti_mask, dicom_images, dicom_affine):

    # Get the affine from the NIfTI mask
    nifti_affine = nifti_mask.affine

    # Compute the transformation matrix from NIfTI to DICOM
    transform_affine = np.linalg.inv(dicom_affine) @ nifti_affine

    # Resample the mask using affine transformation
    nifti_data = nifti_mask.get_fdata()
    # Transpose the mask shape to match channel order
    nifti_data = np.transpose(nifti_data, (2, 0, 1))
    dicom_shape = (len(dicom_images), dicom_images[0].Rows, dicom_images[0].Columns)  # DICOM shape (z, y, x)
    mask_resampled = affine_transform(
        nifti_data,
        np.linalg.inv(transform_affine),
        output_shape=dicom_shape,
        order=0  # Nearest neighbor interpolation to preserve 0/1 values
    )
    # # Transpose the mask shape to match channel order
    # mask_resampled = np.transpose(mask_resampled, (1, 2, 0))

     # Ensure binary values (rounding to 0 or 1)
    mask_resampled = np.round(mask_resampled).astype(np.uint8)

    return mask_resampled

def get_affine_matrix_from_dicom(dcm_series_path):

    # Get the DICOM series as a 3D volume
    dicom_files = sorted(os.listdir(dcm_series_path))
    dicom_images = [pydicom.dcmread(os.path.join(dcm_series_path, f)) for f in dicom_files]
    dicom_images.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Extract DICOM orientation, spacing, and origin
    first_dicom = dicom_images[0]
    dicom_orientation = np.array(first_dicom.ImageOrientationPatient).reshape(2, 3)  # (2, 3)
    dicom_spacing = np.array(list(first_dicom.PixelSpacing) + [first_dicom.SliceThickness])  # (3,)
    dicom_origin = np.array(first_dicom.ImagePositionPatient)  # (3,)
    
    # Ensure dicom_orientation is a full 3x3 matrix
    dicom_orientation = np.vstack([dicom_orientation, np.cross(dicom_orientation[0], dicom_orientation[1])])

    # Build the DICOM affine matrix
    dicom_affine = np.eye(4)
    dicom_affine[:3, :3] = dicom_orientation * dicom_spacing[:, None]  # Scale orientation by spacing
    dicom_affine[:3, 3] = dicom_origin

    # Convert from DICOM LPI to NIfTI RPS
    # Flip the x-axis (L->R) and z-axis (S->P), keeping the y-axis unchanged
    lpi_to_rps = np.diag([-1, 1, 1, 1])  # Flip x (L->R), keep y (P->P), flip z (S->P)
    rps_affine = lpi_to_rps @ dicom_affine

    return rps_affine

def resample_image(image, new_spacing=None, new_size=None,
                   interpolator=sitk.sitkBSpline, tol=0.00001):
    """Resample an image to another spacing.

    Parameters
    ----------
    image : ITK Image
        Input image.
    new_spacing : list
        Spacing to resample image to

    Returns
    -------
    resampled_image : ITK Image
        Output image.
    """
    if new_spacing is not None and new_size is not None:
        raise ValueError('Either provide resample_spacing OR resample_size as input!')

    if new_spacing is None and new_size is None:
        raise ValueError('Either provide resample_spacing OR resample_size as input!')

    # Get original settings
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
   
    # ITK can only do 3D images
    if len(original_size) == 2:
        original_size = original_size + (1, )
    if len(original_spacing) == 2:
        original_spacing = original_spacing + (1.0, )

    if new_size is None:
        # Compute output size
        new_size = [round(original_size[0]*(original_spacing[0] + tol) / new_spacing[0]),
                    round(original_size[1]*(original_spacing[1] + tol) / new_spacing[1]),
                    round(original_size[2]*(original_spacing[2] + tol) / new_spacing[2])]

    if new_spacing is None:
        # Compute output spacing
        tol = 0
        new_spacing = [original_size[0]*(original_spacing[0] + tol)/new_size[0],
                       original_size[1]*(original_spacing[1] + tol)/new_size[1],
                       original_size[2]*(original_spacing[2] + tol)/new_size[2]]

    # Set and execute the filter
    ResampleFilter = sitk.ResampleImageFilter()
    ResampleFilter.SetInterpolator(interpolator)
    ResampleFilter.SetOutputSpacing(new_spacing)
    ResampleFilter.SetSize(np.array(new_size, dtype='int').tolist())
    ResampleFilter.SetOutputDirection(image.GetDirection())
    ResampleFilter.SetOutputOrigin(image.GetOrigin())
    ResampleFilter.SetOutputPixelType(image.GetPixelID())
    ResampleFilter.SetTransform(sitk.Transform())
    try:
        resampled_image = ResampleFilter.Execute(image)
    except RuntimeError:
        # Assume the error is due to the direction determinant being 0
        # Crude solution: simply set a correct direction
        print('[Segmentix Warning] Bad output direction in resampling, resetting direction.')
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ResampleFilter.SetOutputDirection(direction)
        image.SetDirection(direction)
        resampled_image = ResampleFilter.Execute(image)

    return resampled_image