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
images_path = r'H:\Data\Local_Recurrence_Exports'
base_data_path = r'\\mymdafiles\di_data1\Morfeus\bmanderson\Modular_projects\Liver_Local_Recurrence_Work'
excel_file = os.path.join(base_data_path,'Post_treatment_and_Recurrence_info.xlsx')
output_file = os.path.join(base_data_path,'Post_treatment_and_Recurrence_info_output.xlsx')
status_path = os.path.join(base_data_path,'Status')
if not os.path.exists(status_path):
    os.makedirs(status_path)
data = pd.read_excel(output_file)
MRNs = data['MRN']
ablation_volume = []
ablation_recurrence_volume = []
while True:
    for index in range(len(MRNs)):
        MRN = str(data['MRN'][index])
        Recurrence = data['Recurrence'][index]
        # if os.path.exists(os.path.join(status_path, MRN+'.txt')):
        #     continue
        print(MRN)
        patient_path = os.path.join(images_path, MRN)
        if not os.path.exists(patient_path):
            print('Does not exist...')
            continue
        case = os.listdir(patient_path)[0]
        recurrence_path = os.path.join(patient_path, case, Recurrence)
        output_dir = os.path.join(recurrence_path, 'new_RT')
        RS_files = []
        if os.path.exists(output_dir):
            RS_files = [i for i in os.listdir(output_dir) if i.endswith('.dcm')]
        recurrence_reader = DicomReaderWriter(arg_max=False, Contour_Names=['Liver', 'Recurrence',
                                                                            'Ablation_Recurrence', 'Liver_Ablation',
                                                                            'Ablation', 'GTV_Exp_5mm_outside_Ablation'])
        try:
            recurrence_reader.Make_Contour_From_directory(recurrence_path)
        except:
            continue
        spacing = recurrence_reader.annotation_handle.GetSpacing()
        mask = recurrence_reader.mask
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
        if not RS_files:
            centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_recurrence))
            centroid_of_ablation = np.asarray(center_of_mass(ablation))
            output_recurrence = create_output_ray(centroid_of_ablation_recurrence, ref_binary_image=recurrence_base,
                                                  spacing=spacing, min_max_only=False, target_centroid=centroid_of_ablation)
            recurrence_reader.with_annotations(output_recurrence, output_dir=output_dir,
                                               ROI_Names=['cone_recurrence', 'cone_projected'])
        else:
            cone_reader = DicomReaderWriter(arg_max=False, Contour_Names=['cone_recurrence', 'cone_projected'])
            cone_reader.make_array(recurrence_path)
            cone_reader.rois_in_case = []
            cone_reader.RTs_in_case = {}
            cone_reader.add_RT(output_dir)
            cone_reader.get_mask()
            output_recurrence = cone_reader.mask
        overlap = np.where((output_recurrence[..., -1] == 1) & (min_ablation_margin == 1)) # See if it overlaps with the minimum ablation margin
        if np.any(overlap):
            volume_overlap = len(overlap[0])*np.prod(spacing)/1000  # cm^3
            data['Overlap?'][index] = 1.0
        else:
            volume_overlap = 0
            data['Overlap?'][index] = 0.0
        data['Volume (cc)'][index] = volume_overlap
        print('Overlap volume is {} for {}'.format(volume_overlap, MRN))
        data.to_excel(output_file, index=0)
        fid = open(os.path.join(status_path, MRN+'.txt'), 'w+')
        fid.close()
    time.sleep(10)