__author__ = 'Brian M Anderson'
# Created on 1/2/2020
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
from Utilities import *
from skimage import morphology


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
    labels = morphology.label(cross_base, neighbors=4)  # Could have multiple recurrence sites
    output = np.zeros(cross_base.shape)
    output_recurrence = np.expand_dims(output, axis=-1)
    output_recurrence = np.repeat(output_recurrence, repeats=2, axis=-1)
    for label_value in range(1, np.max(labels) + 1):
        recurrence = np.zeros(cross_base.shape)
        recurrence[labels == label_value] = 1
        polar_cords = create_distance_field(recurrence, origin=centroid_of_ablation_recurrence, spacing=spacing)
        polar_cords = np.round(polar_cords, 3).astype('float16')

        polar_cords = polar_cords[:, 1:]
        '''
        We now have the min/max phi/theta for pointing the recurrence_ablation site to the recurrence

        Now, we take those coordinates and see if, with the ablation to minimum ablation site overlap

        Note: This will turn a star shape into a square which encompasses the star!
        '''
        output_recurrence[..., 1] += define_cone(polar_cords, centroid_of_ablation_recurrence, cross_base,
                                                 spacing, margin=75, min_max=False, margin_rad=np.deg2rad(5))
    recurrence_reader.with_annotations(output_recurrence, output_dir=os.path.join(recurrence_path, 'new_RT'),
                                       ROI_Names=['cone_cross_fixed'])