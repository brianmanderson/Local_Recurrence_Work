__author__ = 'Brian M Anderson'
# Created on 11/10/2020
import os
from connect import *
import time, getpass
import pandas as pd
import numpy as np


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    df = df.loc[(df['Registered'] == 0) & (df['Has_Disease_Seg'] == 0)]
    MRN_list, GTV_List, Ablation_list, Case_list = df['MRN'].values, df['PreExam'].values, df['Ablation_Exam'].values, df['Case'].values
    MRN_dictionary = {}
    for MRN, GTV, Ablation, case in zip(MRN_list, GTV_List, Ablation_list, Case_list):
        if MRN not in MRN_dictionary:
            MRN_dictionary[MRN] = {case: []}
        for exam in [GTV, Ablation]:
            if type(exam) is not float:
                exam = str(exam)
                if exam.startswith('CT'):
                    if exam.find(' ') == -1:
                        exam = 'CT {}'.format(exam.split('CT')[-1])
                    if exam not in MRN_dictionary[MRN][case]:
                        MRN_dictionary[MRN][case].append(exam)
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
    current_MRN = None
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        print(MRN)
        for case_num in MRN_dictionary[MRN_key]:
            for exam_name in MRN_dictionary[MRN_key][case_num]:
                export_path = os.path.join(base_export_path, MRN, 'Case {}'.format(case_num), exam_name)
                if os.path.exists(export_path) and os.listdir(export_path):
                    dicom_files = [i for i in os.listdir(export_path) if i.endswith('.dcm')]
                    if dicom_files:
                        continue  # Path already exists and has files
                if not os.path.exists(export_path):
                    os.makedirs(export_path)
                if current_MRN != MRN:
                    try:
                        class_struct.ChangePatient(MRN)
                        current_MRN = MRN
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
                try:
                    exam = case.Examinations[exam_name]
                except:
                    continue
                case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[exam.Name],
                                           RtStructureSetsForExaminations=[exam.Name])
            else:
                fid = open(os.path.join(status_path, '{}_None.txt'.format(MRN)), 'w+')
                fid.close()


if __name__ == "__main__":
    main()
