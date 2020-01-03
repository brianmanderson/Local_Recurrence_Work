__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from scipy.ndimage.measurements import center_of_mass
from Utilities import *


'''
This should have two parts... first, check the recurrence image for what direction
the recurrence occurred
Then, look at the post-treatment image and see if there was 5 mm margin existing in that direction
'''
images_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
excel_file = os.path.join('..','Data','Post_treatment_and_Recurrence_info.xlsx')
output_file = os.path.join('..','Data','Post_treatment_and_Recurrence_info_output.xlsx')
status_path = os.path.join('..','Data','Status')
if not os.path.exists(status_path):
    os.makedirs(status_path)
data = pd.read_excel(excel_file)
MRNs = data['MRN']
for index in range(len(MRNs)):
    MRN = str(data['MRN'][index])
    print(MRN)
    Recurrence = data['Recurrence'][index]
    if os.path.exists(os.path.join(status_path,MRN+'.txt')):
        continue
    recurrence_path = os.path.join(images_path,MRN,Recurrence)
    recurrence_reader = Dicom_to_Imagestack(arg_max=False,Contour_Names=['Liver','Recurrence','Ablation_Recurrence',
                                                                         'Liver_Ablation','Ablation','GTV_Exp_5mm_outside_Ablation'])
    recurrence_reader.Make_Contour_From_directory(recurrence_path)

    mask = recurrence_reader.mask
    liver_recurrence = mask[...,1]
    ablation_recurrence = mask[...,3]
    recurrence_base = mask[...,2]
    ablation_recurrence[liver_recurrence==0] = 0
    recurrence_base[liver_recurrence==0] = 0

    liver_ablation = mask[..., 4]
    ablation = mask[..., 5]
    min_ablation_margin = mask[..., 6]
    ablation[liver_ablation == 0] == 0
    min_ablation_margin[liver_ablation == 0] = 0


    centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_recurrence))
    centroid_of_ablation = np.asarray(center_of_mass(ablation))
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    output_recurrence = create_output_ray(centroid_of_ablation_recurrence,ref_binary_image=recurrence_base,
                                          spacing=spacing, min_max=True, target_centroid=centroid_of_ablation)
    overlap = np.where((output_recurrence[...,-1]==1) & (min_ablation_margin==1)) # See if it overlaps with the minimum ablation margin
    if overlap:
        data['Overlap?'][index] = 1.0
    else:
        data['Overlap?'][index] = 0.0
    recurrence_reader.with_annotations(output_recurrence, output_dir=os.path.join(recurrence_path,'new_RT'),
                                       ROI_Names=['cone_recurrence', 'cone_projected'])
    data.to_excel(output_file)
    fid = open(os.path.join(status_path,MRN+'.txt'),'w+')
    fid.close()