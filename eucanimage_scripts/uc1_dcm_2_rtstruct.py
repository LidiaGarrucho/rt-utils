import os
import sys
import pydicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path

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
    output_labels = '/mnt/sdc1/EUCanImage/UC1/test_case_4_cmrad/labels_KAUNO.csv'
    output_path = '/mnt/sdc1/EUCanImage/UC1/test_case_4_cmrad/rt_struct_test'

    patient_ids = ['ECI_KAU_S0002']
    auto_seg_folder = '/media/lidia/data2/eucanimage_dimitri/sub_zscored_resampled/val_split/output_107_1k'
    dcm_paths = ['/mnt/sdc1/EUCanImage/UC1/test_case_4_cmrad/exp_ECI_KAU_S0002_20210602/scans/19-t1_vibe_fs_tra_bh_po_DELAYED/resources/DICOM/files'
                 ]
    seg_paths = ['/mnt/sdc1/EUCanImage/UC1/test_case_4_cmrad/exp_ECI_KAU_S0002_20210602/scans/19-t1_vibe_fs_tra_bh_po_DELAYED/resources/annotations/files/event_caac41a2-79d9-43f1-8aa8-150aded18a4d/segmentation.dcm'
                 ]

    auto_seg_paths = ['/mnt/sdc1/EUCanImage/UC1/test_case_4_cmrad/automatic_euc_0280.nii.gz'
                    ]
    
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
        # Transpose dicom_slices
        dicom_slices = np.transpose(dicom_slices, axes=(2, 1, 0))  # (height, width, num_slices) -> (width, height, num_slices)

        # Step 3: Extract pixel spacing and slice thickness
        pixel_spacing = [
            float(dicom_images[0].PixelSpacing[0]),
            float(dicom_images[0].PixelSpacing[1]),
        ]
        slice_thickness = float(dicom_images[0].SliceThickness)
        spacing = [pixel_spacing[0], pixel_spacing[1], slice_thickness]

        # Step 4: Construct the affine transformation matrix
        origin = np.array(dicom_images[0].ImagePositionPatient, dtype=np.float32)
        orientation = np.array(dicom_images[0].ImageOrientationPatient, dtype=np.float32).reshape(2, 3)
        orientation = np.vstack([orientation, np.cross(orientation[0], orientation[1])])  # Add third direction

        # Adjust affine matrix for RAI orientation
        orientation[0] *= -1  # Flip x-axis to align with RAI
        orientation[1] *= -1  # Flip y-axis to align with RAI
        dicom_affine = np.eye(4)
        dicom_affine[:3, :3] = orientation * spacing  # Scale by voxel spacing
        dicom_affine[:3, 3] = origin  # Set the origin

        dicom_nifti = nib.Nifti1Image(dicom_slices, dicom_affine)
        dicom_nifti_path = os.path.join(output_path_patient, f"{patient_id}.nii.gz")
        if save_debug_files:
            nib.save(dicom_nifti, dicom_nifti_path)
            print(f"DICOM (RPI) volume saved to: {dicom_nifti_path}")

        rtstruct = RTStructBuilder.create_from(
                                dicom_series_path=dcm_series_path, 
                                rt_struct_path=seg_path)
        if save_debug_files:
            # View all of the ROI names from within the image
            print(rtstruct.get_roi_names())

        # If ROI Name is not L1, delete it
        for mask_name in rtstruct.get_roi_names():
            if mask_name != 'L1':
                rtstruct.del_roi_name(mask_name)
        if len(rtstruct.get_roi_names()) > 1:
            print(f"More than one L1 ROI found: {rtstruct.get_roi_names()} - {patient_id}")

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
        rtstruct.modify_roi_name_and_color('L1', new_roi_name, new_roi_color)

        roi_names = rtstruct.get_roi_names()
        if save_debug_files:
            for roi_name in roi_names:
                # Loading the 3D Mask from within the RT Struct
                mask_3d = np.uint8(rtstruct.get_roi_mask_by_name(roi_name))
                # mask_3d = np.transpose(mask_3d, axes=(2, 1, 0))
                # mask_3d = np.rot90(mask_3d, k=1, axes=(0, 1))
                # mask_3d = np.transpose(mask_3d, (1, 0, 2))
                # Reverse the z-axis flip
                # mask_3d = mask_3d[:, :, ::-1]
                # Reverse the 90-degree rotation (counterclockwise rotation to undo the previous 270-degree rotation)
                # mask_3d = np.rot90(mask_3d, k=2, axes=(0, 1))  # 90 degrees counterclockwise
                # mask_3d = mask_3d[:, :, ::-1]
                # Create a NIfTI image
                nifti_image = nib.Nifti1Image(mask_3d, dicom_nifti.affine, dicom_nifti.header)
                # Save the NIfTI image
                nib.save(nifti_image, os.path.join(output_path_patient, patient_id + f"_manual_original_{new_roi_name}.nii.gz"))
                print(f"Saved NIfTI mask file for {patient_id}")

        # for mask_name in rtstruct.get_roi_names():
        #     rtstruct.del_roi_name(mask_name)
        
        # Read the mask from the NIfTI file
        # Load NIfTI mask and adjust orientation
        nifti_mask = nib.load(auto_seg_path)
        # Resample the mask using affine transformation
        dicom_shape = (len(dicom_images), dicom_images[0].Rows, dicom_images[0].Columns)  # DICOM shape (z, y, x)
        mask_3d = nifti_mask.get_fdata()
        # mask_3d = np.transpose(mask_3d, (2, 0, 1))
        # Reverse the z-axis flip
        mask_3d = mask_3d[:, :, ::-1]
        # Reverse the 90-degree rotation (counterclockwise rotation to undo the previous 270-degree rotation)
        mask_3d = np.rot90(mask_3d, k=1, axes=(0, 1))  # 90 degrees counterclockwise
        mask_3d = mask_3d[:, :, ::-1]
        sitk_mask = sitk.GetImageFromArray(mask_3d)
        sitk_mask = resample_image(image=sitk_mask, new_size=dicom_shape, interpolator=sitk.sitkNearestNeighbor)
        mask_3d = sitk.GetArrayFromImage(sitk_mask)

        # Turn the pixel array into a 3D numpy array of booleans
        mask_3d = mask_3d.astype(bool)

        # Add another ROI, this time setting the color, description, and name
        rtstruct.add_roi(
                        mask=mask_3d, 
                        color=auto_roi_color, # RGB color  yellow: [255, 255, 0], red: [255, 0, 0]
                        name=auto_roi_name)
        if save_debug_files:
            rtstruct.save(os.path.join(output_path_patient, f'{patient_id}_rt_struct.dcm'))
        else:
            rtstruct.save(os.path.join(output_path, f'{patient_id}_segmentation.dcm'))

        if save_debug_files:
            rtstruct = RTStructBuilder.create_from(
                                    dicom_series_path=dcm_series_path, 
                                    rt_struct_path=os.path.join(output_path_patient, f'{patient_id}_rt_struct.dcm'))
            # View all of the ROI names from within the image
            print(rtstruct.get_roi_names())

            for mask_name in rtstruct.get_roi_names():
                # Loading the 3D Mask from within the RT Struct
                mask_3d = np.uint8(rtstruct.get_roi_mask_by_name(mask_name))
                mask_3d = np.rot90(mask_3d, k=3, axes=(0, 1))
                mask_nifti = nib.Nifti1Image(mask_3d.astype(np.uint8), dicom_affine)
                mask_nifti_path = os.path.join(output_path_patient, f"{mask_name}_mask_rtstructu2nii.nii.gz")
                nib.save(mask_nifti, mask_nifti_path)
                print(f"Mask saved to: {mask_nifti_path}")
        exit(0)
    # Save the labels
    df_labels.to_csv(output_labels, index=False)
    exit(0)