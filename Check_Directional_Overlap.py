__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
from Utilities import *
from skimage import morphology


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
    recurrence = mask[...,2]
    ablation_recurrence[liver_recurrence==0] = 0
    recurrence[liver_recurrence==0] = 0
    labels = morphology.label(recurrence, neighbors=4) # Could have multiple recurrence sites
    for label in range(1,np.max(labels)+1):
    centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_recurrence))
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    polar_cords = create_distance_field(recurrence,origin=centroid_of_ablation_recurrence, spacing=spacing)
    polar_cords = np.round(polar_cords,3).astype('float16')

    polar_cords = polar_cords[:, 1:]
    '''
    We now have the min/max phi/theta for pointing the recurrence_ablation site to the recurrence
    
    Now, we take those coordinates and see if, with the ablation to minimum ablation site overlap
    
    Note: This will turn a star shape into a square which encompasses the star!
    '''
    output = define_cone(polar_cords, centroid_of_ablation_recurrence, liver_recurrence, spacing, margin=75, min_max=True)
    cone = np.where(output==1)
    output_recurrence = np.expand_dims(output, axis=-1)
    output_recurrence = np.repeat(output_recurrence,repeats=3,axis=-1)

    '''
    Now, define it on the centroid of mapped ablation
    '''
    liver_ablation = mask[...,4]
    ablation = mask[...,5]
    min_ablation_margin = mask[...,6]
    ablation[liver_ablation==0] == 0
    min_ablation_margin[liver_ablation==0] = 0
    centroid_of_ablation = np.asarray(center_of_mass(ablation))
    output = define_cone(polar_cords, centroid_of_ablation, liver_recurrence, spacing, margin=75,min_max=True)
    overlap = np.where((output==1) & (min_ablation_margin==1)) # See if it overlaps with the minimum ablation margin
    if overlap:
        data['Overlap?'][index] = 1.0
    else:
        data['Overlap?'][index] = 0.0
    output_recurrence[...,-1] = output
    recurrence_reader.with_annotations(output_recurrence, output_dir=os.path.join(recurrence_path,'new_RT'),
                                       ROI_Names=['cone_recurrence', 'cone_projected'])
    data.to_excel(output_file)
    fid = open(os.path.join(status_path,MRN+'.txt'),'w+')
    fid.close()