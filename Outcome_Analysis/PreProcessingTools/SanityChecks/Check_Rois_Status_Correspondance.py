__author__ = 'Brian M Anderson'
# Created on 1/22/2021
from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool.\
    ReaderWriter import DicomReaderWriter, plot_scroll_Image, sitk, np
import os
import pandas as pd


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    df = df.loc[(df['Registered'] == 0) & (df['Has_Liver'] == 1) & (df['Has_Disease_Seg'] == 0)]
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


def check_rois(excel_file, dicom_export_path):
    MRN_dictionary = return_MRN_dictionary(excel_path=excel_file)
    df = pd.read_excel(excel_file, sheet_name='Refined')
    df = df.loc[(df['Registered'] == 0) & (df['Has_Disease_Seg'] == 0)]
    primary_reader = DicomReaderWriter(Contour_Names=['Retro_GTV', 'Retro_GTV_Recurred'],
                                       require_all_contours=False, arg_max=False, verbose=False)
    stats = sitk.LabelShapeStatisticsImageFilter()
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
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
            pat_df = df.loc[(df['Registered'] == 0) & (df['Has_Liver'] == 1) & (df['Has_Disease_Seg'] == 0) &
                            (df['MRN'] == MRN_key) & (df['Case'] == int(case_num)) & (df['PreExam'] == primary) &
                            (df['Ablation_Exam'] == secondary)]
            case_path = os.path.join(dicom_export_path, MRN, 'Case {}'.format(case_num))
            if not os.path.exists(case_path):
                continue
            primary_path = os.path.join(case_path, primary)
            if not os.path.exists(primary_path):
                continue
            primary_reader.__reset__()
            primary_reader.walk_through_folders(primary_path)
            primary_reader.get_images_and_mask()
            mask = primary_reader.mask
            '''
            First, check to make sure the number of status = 3 is equal to the number in the mask
            '''
            status_3 = mask[..., 1]
            num_status_3 = 0
            if np.max(status_3) > 0:
                status_3_image = sitk.GetImageFromArray(status_3.astype('int'))
                connected_image = Connected_Component_Filter.Execute(status_3_image)
                stats.Execute(connected_image)
                num_status_3 = len(stats.GetLabels())
            label_3 = pat_df.loc[pat_df['Status'] == 3]
            if label_3.shape[0] != num_status_3:
                print('Values did not match up for status 3 {}. Had {}, needed {}'.format(MRN, num_status_3,
                                                                                          label_3.shape[0]))
            '''
            Next, check to make sure the number of status = 1 or 2 is equal to the number in the mask
            '''
            status_1_or_2 = mask[..., 2]
            num_status_1_or_2 = 0
            if np.max(status_1_or_2) > 0:
                status_3_image = sitk.GetImageFromArray(status_1_or_2.astype('int'))
                connected_image = Connected_Component_Filter.Execute(status_3_image)
                stats.Execute(connected_image)
                num_status_1_or_2 = len(stats.GetLabels())
            label_1_or_2 = pat_df.loc[(pat_df['Status'] == 1) | (pat_df['Status'] == 2)]
            if label_1_or_2.shape[0] != num_status_1_or_2:
                print('Values did not match up for status 1/2 {}. Had {}, needed {}'.format(MRN, num_status_1_or_2,
                                                                                            label_1_or_2.shape[0]))


if __name__ == '__main__':
    pass
