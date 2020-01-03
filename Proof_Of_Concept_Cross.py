__author__ = 'Brian M Anderson'
# Created on 1/2/2020
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
from Utilities import *


images_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
excel_file = os.path.join('..', 'Data', 'Post_treatment_and_Recurrence_info.xlsx')
output_file = os.path.join('..', 'Data', 'Post_treatment_and_Recurrence_info_output.xlsx')
status_path = os.path.join('..', 'Data', 'Status')
if not os.path.exists(status_path):
    os.makedirs(status_path)
data = pd.read_excel(excel_file)
MRNs = data['MRN']
for index in range(len(MRNs)):
    MRN = str(data['MRN'][index])
    print(MRN)
    Recurrence = data['Recurrence'][index]
    if os.path.exists(os.path.join(status_path, MRN + '.txt')):
        continue
    recurrence_path = os.path.join(images_path, MRN, Recurrence)
    recurrence_reader = Dicom_to_Imagestack(arg_max=False, Contour_Names=['Test_Ablation','Test_Cross'])
    recurrence_reader.Make_Contour_From_directory(recurrence_path)

    mask = recurrence_reader.mask
    ablation_base = mask[...,1]
    cross_base = mask[...,2]

    centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_base))
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    output = create_output_ray(centroid_of_ablation_recurrence, spacing=spacing, target_binary_image=cross_base,
                      margin_rad=np.deg2rad(5), margin=50, min_max=False)
    output = np.expand_dims(output, axis=-1)
    output = np.repeat(output, repeats=2, axis=-1)
    recurrence_reader.with_annotations(output, output_dir=os.path.join(recurrence_path, 'new_RT'),
                                       ROI_Names=['cone_cross_fixed'])