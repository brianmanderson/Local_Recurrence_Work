__author__ = 'Brian M Anderson'
# Created on 11/10/2020
import os
from connect import *
import time, getpass
import pandas as pd
import numpy as np


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    MRN_list, status_list = df['MRN'].values, df['Status'].values
    MRN_dictionary = {}
    for MRN, status in zip(MRN_list, status_list):
        if MRN not in MRN_dictionary:
            MRN_dictionary[MRN] = []
        if type(status) is float:
            if status not in MRN_dictionary[MRN]:
                MRN_dictionary[MRN].append(int(status))
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
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        print(MRN)
        try:
            class_struct.ChangePatient(MRN)
        except:
            continue
        if class_struct.patient is None:
            continue
        case = None
        for case in class_struct.patient.Cases:
            continue
        export_pair = False
        '''
        First, check to see if one of the ROIs is present on the pre-treatment exam
        '''
        for exam_name in MRN_dictionary[MRN_key]:
            try:
                exam = case.Examinations[exam_name]
            except:
                continue
            for roi in ['Retro_GTV', 'Retro_Ablation', 'Retro_GTV_Recurred']:
                if case.PatientModel.StructureSets[exam.Name].RoiGeometries[roi].HasContours():
                    '''
                    If it exists, export the pre-treatment and post-treatment scan
                    '''
                    export_pair = True
                    break
        if export_pair:
            for exam_name in MRN_dictionary[MRN_key]:
                try:
                    exam = case.Examinations[exam_name]
                except:
                    continue
                export_path = os.path.join(base_export_path, MRN, case.CaseName, exam_name)
                if os.path.exists(export_path) and os.listdir(export_path):
                    dicom_files = [i for i in os.listdir(export_path) if i.endswith('.dcm')]
                    if dicom_files:
                        continue  # Path already exists and has files
                if not os.path.exists(export_path):
                    os.makedirs(export_path)
                case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[exam.Name],
                                           RtStructureSetsForExaminations=[exam.Name])


if __name__ == "__main__":
    main()
