__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
from Utilities import *



'''
This should have two parts... first, check the recurrence image for what direction
the recurrence occurred
Then, look at the post-treatment image and see if there was 5 mm margin existing in that direction
'''
new_images = np.load(os.path.join('..', 'saved2.npy'))
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
    recurrence_reader = Dicom_to_Imagestack(arg_max=False,Contour_Names=['Liver_recurrence','recurrence','Ablation_Recurrence',
                                                                         'Liver_Ablation','Ablation','GTV_Exp_5mm_outside_Ablation'])
    recurrence_reader.Make_Contour_From_directory(recurrence_path)

    mask = recurrence_reader.mask
    liver_recurrence = mask[...,1]
    ablation_recurrence = mask[...,3]
    recurrence = mask[...,2]
    ablation_recurrence[liver_recurrence==0] = 0
    recurrence[liver_recurrence==0] = 0
    centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_recurrence))
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    polar_cords = create_distance_field(recurrence,origin=centroid_of_ablation_recurrence, spacing=spacing)
    polar_cords = np.round(polar_cords,4)

    min_phi, max_phi, min_theta, max_theta = min(polar_cords[...,1]), max(polar_cords[...,1]), min(polar_cords[...,2]),\
                                             max(polar_cords[...,2])
    '''
    We now have the min/max phi/theta for pointing the recurrence_ablation site to the recurrence
    
    Now, we take those coordinates and see if, with the ablation to minimum ablation site overlap
    
    Note: This will turn a star shape into a square which encompasses the star!
    '''
    cone_cords = create_distance_field(np.ones(liver_recurrence.shape),origin=centroid_of_ablation_recurrence,spacing=spacing)
    cone_cords = np.round(cone_cords,4)
    output = np.zeros(cone_cords.shape[0])
    vals = np.where((cone_cords[:,1]>=min_phi)&(cone_cords[:,1]<=max_phi)&(cone_cords[:,2]>=min_theta)&(cone_cords[:,2]<=max_theta))
    output[vals[0]] = 1
    output_recurrence = np.reshape(output,liver_recurrence.shape) # This is now a cone including the recurrence site
    output_recurrence = np.expand_dims(output_recurrence, axis=-1)
    output_recurrence = np.repeat(output_recurrence,repeats=3,axis=-1)

    liver_ablation = mask[...,4]
    ablation = mask[...,5]
    min_ablation_margin = mask[...,6]
    ablation[liver_ablation==0] == 0
    min_ablation_margin[liver_ablation==0] = 0
    centroid_of_ablation = np.asarray(center_of_mass(ablation))
    cone_cords = create_distance_field(np.ones(liver_ablation.shape),origin=centroid_of_ablation,spacing=spacing)
    cone_cords = np.round(cone_cords,4)
    output = np.zeros(cone_cords.shape[0])
    vals = np.where((cone_cords[:,1]>=min_phi)&(cone_cords[:,1]<=max_phi)&(cone_cords[:,2]>=min_theta)&(cone_cords[:,2]<=max_theta))
    output[vals[0]] = 1
    output = np.reshape(output,liver_ablation.shape) # This is now a cone including the recurrence site
    overlap = np.where((output==1) & (min_ablation_margin==1)) # See if it overlaps with the minimum ablation margin
    if overlap:
        data['Overlap?'][index] = 1.0
    else:
        data['Overlap?'][index] = 0.0
    data.to_excel(output_file)
    output_recurrence[...,-1] = output
    recurrence_reader.with_annotations(output_recurrence, output_dir=os.path.join(recurrence_path,'new_RT'),
                                       ROI_Names=['cone_recurrence', 'cone_projected'])
    fid = open(os.path.join(status_path,MRN+'.txt'),'w+')
    fid.close()