__author__ = 'Brian M Anderson'
# Created on 11/10/2020
import os
try:
    from connect import *
except:
    xxx = 1
import time, getpass
import pandas as pd
import numpy as np


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


class create_RT_Structure():
    def __init__(self):
        self.patient_db = get_current('PatientDB')

    def ChangePatient(self, MRN):
        print('got here')
        self.MRN = MRN
        self.patient = None
        info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=False)
        if not info_all:
            info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=True)
        for info in info_all:
            if info['PatientID'] == self.MRN:
                self.patient = self.patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
                self.MRN = self.patient.PatientID
                return None
        print('did not find a patient')


def main():
    status_path = r'H:\Deeplearning_Recurrence_Work\Status\Export'
    base_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    for path in [status_path, base_export_path]:
        if not os.path.exists(path):
            os.makedirs(path)
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'

    MRN_dictionary = return_MRN_dictionary(excel_path)
    class_struct = create_RT_Structure()
    current_mrn = None
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        print(MRN)
        for primary, secondary, case_num in zip(MRN_dictionary[MRN_key]['Primary'],
                                                MRN_dictionary[MRN_key]['Secondary'],
                                                MRN_dictionary[MRN_key]['Case_Number']):
            export_path = os.path.join(base_export_path, MRN, 'Case {}'.format(case_num),
                                       'Registration_{}_to_{}'.format(primary, secondary))
            if os.path.exists(export_path) and os.listdir(export_path):
                dicom_files = [i for i in os.listdir(export_path) if i.endswith('.dcm') or
                               i.startswith('SameFrameOfReference')]
                if dicom_files:
                    print('Already done {}'.format(MRN))
                    continue  # Path already exists and has files
            if current_mrn != MRN:
                try:
                    class_struct.ChangePatient(MRN)
                    current_mrn = MRN
                except:
                    break
                if class_struct.patient is None:
                    break
            found_case = False
            case = None
            for case in class_struct.patient.Cases:
                if int(case.CaseName.split(' ')[-1]) == int(case_num):
                    found_case = True
                    break
            if not found_case:
                continue
            exam_names = [e.Name for e in case.Examinations]
            if primary not in exam_names or secondary not in exam_names:
                print('Lacking exams')
                continue
            if case.Examinations[primary].EquipmentInfo.FrameOfReference == \
                    case.Examinations[secondary].EquipmentInfo.FrameOfReference:
                if not os.path.exists(export_path):
                    os.makedirs(export_path)
                fid = open(os.path.join(export_path, 'SameFrameOfReference.txt'), 'w+')
                fid.close()
            else:
                for registration in case.Registrations:
                    to_for = registration.ToFrameOfReference
                    # Frame of reference of the "From" examination.
                    from_for = registration.FromFrameOfReference
                    # Find all examinations with frame of reference that matches 'to_for'.
                    to_examinations = [e.Name for e in case.Examinations if e.EquipmentInfo.FrameOfReference == to_for]
                    # Find all examinations with frame of reference that matches 'from_for'.
                    from_examinations = [e.Name for e in case.Examinations if e.EquipmentInfo.FrameOfReference == from_for]
                    if primary in to_examinations and secondary in from_examinations:
                        if not os.path.exists(export_path):
                            os.makedirs(export_path)
                        if registration.RegistrationSource is not None:
                            exam_names = ["%s:%s" % (registration.RegistrationSource.ToExamination.Name,
                                                     registration.RegistrationSource.FromExamination.Name)]
                            case.ScriptableDicomExport(ExportFolderPath=export_path,
                                                       SpatialRegistrationForExaminations=exam_names,
                                                       IgnorePreConditionWarnings=True)
                        else:
                            fid = open(os.path.join(export_path, 'SameFrameOfReference.txt'), 'w+')
                            fid.close()


if __name__ == "__main__":
    main()
