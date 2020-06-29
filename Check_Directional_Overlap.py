__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from scipy.ndimage.measurements import center_of_mass
from Local_Recurrence_Work.Ray_Tracing.Utilities import *
import time


'''
This should have two parts... first, check the recurrence image for what direction
the recurrence occurred
Then, look at the post-treatment image and see if there was 5 mm margin existing in that direction
'''
images_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
base_data_path = r'\\mymdafiles\di_data1\Morfeus\bmanderson\Modular_projects\Liver_Local_Recurrence_Work'
excel_file = os.path.join(base_data_path,'Post_treatment_and_Recurrence_info.xlsx')
output_file = os.path.join(base_data_path,'Post_treatment_and_Recurrence_info_output.xlsx')
status_path = os.path.join(base_data_path,'Status')
if not os.path.exists(status_path):
    os.makedirs(status_path)
data = pd.read_excel(excel_file)
MRNs = data['MRN']
ablation_volume = []
ablation_recurrence_volume = []
while True:
    for index in range(len(MRNs)):
        MRN = str(data['MRN'][index])
        Recurrence = data['Recurrence'][index]
        if os.path.exists(os.path.join(status_path,MRN+'.txt')):
            continue
        print(MRN)
        if not os.path.exists(os.path.join(status_path,MRN+'_go.txt')):
            continue
        recurrence_path = os.path.join(images_path,MRN,Recurrence)
        recurrence_reader = Dicom_to_Imagestack(arg_max=False,Contour_Names=['Liver','Recurrence','Ablation_Recurrence',
                                                                             'Liver_Ablation','Ablation','GTV_Exp_5mm_outside_Ablation'])
        try:
            recurrence_reader.Make_Contour_From_directory(recurrence_path)
        except:
            continue

        mask = recurrence_reader.mask
        bounds = [0, mask.shape[0]]
        # mask = mask[130:160,...]
        liver_recurrence = mask[..., 1]
        ablation_recurrence = mask[..., 3]
        recurrence_base = mask[..., 2]
        ablation_recurrence[liver_recurrence == 0] = 0
        recurrence_base[liver_recurrence == 0] = 0

        liver_ablation = mask[..., 4]
        ablation = mask[..., 5]
        min_ablation_margin = mask[..., 6]
        ablation[liver_ablation == 0] == 0
        min_ablation_margin[liver_ablation == 0] = 0

        # ablation_volume.append(np.prod(recurrence_reader.annotation_handle.GetSpacing()) * np.sum(ablation==1))
        # ablation_recurrence_volume.append(np.prod(recurrence_reader.annotation_handle.GetSpacing()) * np.sum(ablation_recurrence==1))
        centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_recurrence))
        centroid_of_ablation = np.asarray(center_of_mass(ablation))
        spacing = recurrence_reader.annotation_handle.GetSpacing()
        output_recurrence = create_output_ray(centroid_of_ablation_recurrence,ref_binary_image=recurrence_base,
                                              spacing=spacing, min_max_only=False, target_centroid=centroid_of_ablation)
        overlap = np.where((output_recurrence[..., -1] == 1) & (min_ablation_margin == 1)) # See if it overlaps with the minimum ablation margin
        if overlap:
            volume_overlap = len(overlap[0])*np.prod(spacing)/1000  # cm^3
            data['Overlap?'][index] = 1.0
        else:
            data['Overlap?'][index] = 0.0
        recurrence_reader.with_annotations(output_recurrence, output_dir=os.path.join(recurrence_path,'new_RT'),
                                           ROI_Names=['cone_recurrence', 'cone_projected'])
        data.to_excel(output_file)
        fid = open(os.path.join(status_path,MRN+'.txt'),'w+')
        fid.close()
    print('sleeping...')
    time.sleep(10)