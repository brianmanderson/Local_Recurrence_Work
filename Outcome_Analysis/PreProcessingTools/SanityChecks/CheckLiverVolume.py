__author__ = 'Brian M Anderson'
# Created on 1/22/2021
from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool.\
    ReaderWriter import DicomReaderWriter, plot_scroll_Image, sitk, np
import os
import pandas as pd


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    df = df.loc[(df['Registered'] == 0) & (df['Has_Disease_Seg'] == 0)]
    MRN_list, primary_list, secondary_list, case_list = df['MRN'].values, df['PreExam'].values,\
                                                        df['Ablation_Exam'].values, df['Case'].values
    MRN_dictionary = {}
    for MRN, primary, secondary, case in zip(MRN_list, primary_list, secondary_list, case_list):
        if MRN in MRN_dictionary:
            pat_dict = MRN_dictionary[MRN]
        else:
            pat_dict = {'Primary': [], 'Secondary': [], 'Case_Number': []}
        if type(primary) is not float:
            primary = str(primary)
            if primary.startswith('CT'):
                if primary.find(' ') == -1:
                    primary = 'CT {}'.format(primary.split('CT')[-1])
        else:
            continue
        if type(secondary) is not float:
            secondary = str(secondary)
            if secondary.startswith('CT'):
                if secondary.find(' ') == -1:
                    secondary = 'CT {}'.format(secondary.split('CT')[-1])
        else:
            continue
        if not primary.startswith('CT') or not secondary.startswith('CT'):
            continue
        if primary not in pat_dict['Primary'] or secondary not in pat_dict['Secondary']:
            pat_dict['Primary'].append(primary)
            pat_dict['Secondary'].append(secondary)
            pat_dict['Case_Number'].append(case)
        MRN_dictionary[MRN] = pat_dict
    return MRN_dictionary


def check_liver_volume(excel_file, dicom_export_path):
    MRN_dictionary = return_MRN_dictionary(excel_path=excel_file)
    out_df = {'MRN': [], 'Primary': [], 'Secondary': [], 'Percentage_Change': []}
    associations = {'Liver_BMA_Program_4': 'Liver_BMA_Program_4'}
    primary_reader = DicomReaderWriter(Contour_Names=['Liver_BMA_Program_4'], associations=associations,
                                       require_all_contours=False, arg_max=False, verbose=False)
    secondary_reader = DicomReaderWriter(Contour_Names=['Liver_BMA_Program_4'], associations=associations,
                                         require_all_contours=False, arg_max=False, verbose=False)
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        if MRN not in os.listdir(dicom_export_path):
            print('Did not have {}'.format(MRN))
            continue
        print('Running {}'.format(MRN))
        for primary, secondary, case_num in zip(MRN_dictionary[MRN_key]['Primary'],
                                                MRN_dictionary[MRN_key]['Secondary'],
                                                MRN_dictionary[MRN_key]['Case_Number']):
            case_path = os.path.join(dicom_export_path, MRN, 'Case {}'.format(case_num))
            if not os.path.exists(case_path):
                continue
            primary_path = os.path.join(case_path, primary)
            secondary_path = os.path.join(case_path, secondary)
            if not os.path.exists(primary_path) or not os.path.exists(secondary_path):
                continue
            primary_reader.__reset__()
            primary_reader.walk_through_folders(primary_path)
            primary_reader.get_images_and_mask()
            secondary_reader.__reset__()
            secondary_reader.walk_through_folders(secondary_path)
            secondary_reader.get_images_and_mask()
            primary_mask = primary_reader.mask
            secondary_mask = secondary_reader.mask
            if np.max(primary_mask[..., 1]) != 1 or np.max(secondary_mask[..., 1]) != 1:
                print('Lacking liver here... {}'.format(MRN))
            liver_primary_volume = np.prod(primary_reader.dicom_handle.GetSpacing()) * np.sum(primary_mask[..., 1])
            liver_secondary_volume = np.prod(secondary_reader.dicom_handle.GetSpacing()) * np.sum(secondary_mask[..., 1])
            difference_volume = np.abs(liver_secondary_volume - liver_primary_volume) / liver_primary_volume * 100
            print('{} had a volume dif of {}'.format(MRN, np.round(difference_volume, 2)))
            out_df['MRN'].append(MRN)
            out_df['Primary'].append(primary)
            out_df['Secondary'].append(secondary)
            out_df['Percentage_Change'].append(difference_volume)
    df = pd.DataFrame(out_df)
    df.to_excel(os.path.join('.', 'volumes.xlsx'), index=0)
    return None


if __name__ == '__main__':
    pass
