import os
import sys
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
from nibabel.processing import resample_from_to

# Set root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_affine_matrix_from_dicom, resample_image
from rt_utils import RTStructBuilder

# Clone repository: 
# git clone https://github.com/LidiaGarrucho/rt-utils.git
# Run Installation:
# pip install -r requirements.txt

if __name__ == "__main__":

    save_debug_files = True
    # Isotropic automatic segmentations
    output_path = '/mnt/sdc1/EUCanImage/UC3/ECI_GUM_S1664_export/rt_struct_test'

    patient_ids = ['ECI_GUM_S1664']

    dcm_paths = ['/mnt/sdc1/EUCanImage/UC3/ECI_GUM_S1664_export/BMIAXNAT_E97008/scans/9/resources/1790291/files'
                 ]
    seg_paths = ['/mnt/sdc1/EUCanImage/UC3/ECI_GUM_S1664_export/BMIAXNAT_E97008/scans/9/resources/1895205/files/event_a8895bc4-cf47-496f-8965-7087da335f9b/segmentation.dcm'
                 ]
    auto_seg_paths = ['/mnt/sdc1/EUCanImage/UC3/ECI_GUM_S1664_export/prediction_uncropped.nii.gz']
    
    np.random.seed(42)
    
    df_labels = pd.DataFrame(columns=['patient_id', 'manual_roi', 'auto_roi'])
    for idx, (dcm_series_path, seg_path, auto_seg_path, patient_id) in enumerate(zip(dcm_paths, seg_paths, auto_seg_paths, patient_ids)):

        output_path_patient = os.path.join(output_path, patient_id)
        if save_debug_files:
            os.makedirs(output_path_patient, exist_ok=True)

        # Get the DICOM series as a 3D volume
        dicom_files = sorted(os.listdir(dcm_series_path))
        dicom_images = [pydicom.dcmread(os.path.join(dcm_series_path, f)) for f in dicom_files]

        # Step 2: Sort slices based on Image Position Patient (z-coordinate)
        dicom_images.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        dicom_slices = np.stack([dcm.pixel_array for dcm in dicom_images], axis=0)
        dicom_slices = np.transpose(dicom_slices, axes=(2, 1, 0))  # (height, width, num_slices) -> (width, height, num_slices)

        # Extract DICOM orientation, spacing, and origin
        first_dicom = dicom_images[0]
        dicom_orientation = np.array(first_dicom.ImageOrientationPatient).reshape(2, 3)  # (2, 3)
        dicom_spacing = np.array(list(first_dicom.PixelSpacing) + [first_dicom.SliceThickness])  # (3,)
        dicom_origin = np.array(first_dicom.ImagePositionPatient)  # (3,)
        dicom_orientation = np.vstack([dicom_orientation, np.cross(dicom_orientation[0], dicom_orientation[1])])  # Full 3x3
        # Build the DICOM affine matrix
        dicom_affine = np.eye(4)
        dicom_affine[:3, :3] = dicom_orientation * dicom_spacing[:, None]  # Scale orientation by spacing
        dicom_affine[:3, 3] = dicom_origin  # Set the origin
        # Convert from DICOM LPS to NIfTI RAI
        lps_to_rai = np.diag([-1, -1, 1, 1])  # Flip x (L->R) and y (P->A), keep z (S->I)
        dicom_affine = lps_to_rai @ dicom_affine

        dicom_nifti = nib.Nifti1Image(dicom_slices, dicom_affine)
        dicom_nifti_path = os.path.join(output_path_patient, f"{patient_id}.nii.gz")
        nib.save(dicom_nifti, dicom_nifti_path)

        rtstruct = RTStructBuilder.create_from(
                                dicom_series_path=dcm_series_path, 
                                rt_struct_path=seg_path)
        if save_debug_files:
            # View all of the ROI names from within the image
            print(rtstruct.get_roi_names())

        # If ROI Name is not L1, delete it
        # for mask_name in rtstruct.get_roi_names():
        #     if mask_name != 'L1':
        #         rtstruct.del_roi_name(mask_name)

        if len(rtstruct.get_roi_names()) > 1:
            print(f"More than one segmentation found: {rtstruct.get_roi_names()} - {patient_id}")
            exit(0)

        if save_debug_files:
            print(f'{rtstruct.get_roi_names()}-{patient_id}')
        # Get a 0 or a 1 randomnly:
        random_number = np.random.rand()
        if random_number > 0.5:
            new_roi_name = 'red'
            new_roi_color = [255, 0, 0]
            auto_roi_name = 'yellow'
            auto_roi_color = [255, 255, 0]
        else:
            new_roi_name = 'yellow'
            new_roi_color = [255, 255, 0]
            auto_roi_name = 'red'
            auto_roi_color = [255, 0, 0]
        
        df_labels.loc[idx] = [patient_id, new_roi_name, auto_roi_name]
        rtstruct.modify_roi_name_and_color(rtstruct.get_roi_names()[0], new_roi_name, new_roi_color)

        roi_names = rtstruct.get_roi_names()
        if save_debug_files:
            for roi_name in roi_names:
                # Loading the 3D Mask from within the RT Struct
                mask_3d = np.uint8(rtstruct.get_roi_mask_by_name(roi_name))
                mask_3d = np.transpose(mask_3d, (1, 0, 2))
                mask_nifti = nib.Nifti1Image(mask_3d, dicom_affine)
                mask_nifti = resample_from_to(mask_nifti, dicom_nifti)

                # Save the NIfTI image
                nib.save(mask_nifti, os.path.join(output_path_patient, f"manual_{roi_name}.nii.gz"))
                print(f"Saved NIfTI mask file for {patient_id}")

        # Delete any ROIs if needed
        # for mask_name in rtstruct.get_roi_names():
        #     rtstruct.del_roi_name(mask_name)
        
        # Load the prediction NIfTI mask and resample to orgiginal DICOM shape
        sitk_image = sitk.ReadImage(dicom_nifti_path)
        sitk_mask = sitk.ReadImage(auto_seg_path)
        dicom_shape = (dicom_images[0].Rows, dicom_images[0].Columns, len(dicom_images))  # DICOM shape (z, y, x)
        sitk_mask = resample_image(image=sitk_mask, new_size=dicom_shape, interpolator=sitk.sitkNearestNeighbor)
        sitk_mask.CopyInformation(sitk_image)
        # Save the resampled mask
        if save_debug_files:
            mask_nifti_path = os.path.join(output_path_patient, f"prediction_{auto_roi_name}_mask.nii.gz")
            sitk.WriteImage(sitk_mask, mask_nifti_path)

        # Load mask using nibabel
        mask_nifti = nib.load(mask_nifti_path)
        mask_3d = mask_nifti.get_fdata()

        # Turn the pixel array into a 3D numpy array of booleans
        mask_3d = mask_3d.astype(bool)

        # Add another ROI, this time setting the color, description, and name
        rtstruct.add_roi(
                        mask=mask_3d, 
                        color=auto_roi_color, # RGB color  yellow: [255, 255, 0], red: [255, 0, 0]
                        name=auto_roi_name)
        if save_debug_files:
            rtstruct.save(os.path.join(output_path_patient, f'{patient_id}_rtstruct.dcm'))
        else:
            rtstruct.save(os.path.join(output_path, f'{patient_id}_segmentation.dcm'))

        if save_debug_files:
            rtstruct = RTStructBuilder.create_from(
                                    dicom_series_path=dcm_series_path, 
                                    rt_struct_path=os.path.join(output_path_patient, f'{patient_id}_rtstruct.dcm'))
            # View all of the ROI names from within the image
            print(rtstruct.get_roi_names())

            for mask_name in rtstruct.get_roi_names():
                # Loading the 3D Mask from within the RT Struct
                mask_3d = np.uint8(rtstruct.get_roi_mask_by_name(mask_name))
                mask_3d = np.transpose(mask_3d, (1, 0, 2))
                mask_nifti = nib.Nifti1Image(mask_3d, dicom_affine)
                mask_nifti = resample_from_to(mask_nifti, dicom_nifti)
                mask_nifti_path = os.path.join(output_path_patient, f"{mask_name}_mask_rtstructu2nii.nii.gz")
                nib.save(mask_nifti, mask_nifti_path)
                print(f"Mask saved to: {mask_nifti_path}")
        exit(0)
    # Save the labels
    df_labels.to_csv(f'{output_path}/labels_trial_UC3.csv', index=False)
    exit(0)