__author__ = 'Brian M Anderson'
# Created on 10/28/2020
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
    status_path = r'H:\Deeplearning_Recurrence_Work\Status\ROIs'
    path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
           r'\Predicting_Recurrence'

    excel_path = os.path.join(path, 'RetroAblation.xlsx')
    MRN_dictionary = return_MRN_dictionary(excel_path)
    class_struct = create_RT_Structure()
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        print(MRN)
        if os.path.exists(os.path.join(status_path, 'Created ROIs_{}.txt'.format(MRN))):
            continue
        try:
            class_struct.ChangePatient(MRN)
        except:
            continue
        if class_struct.patient is None:
            continue
        for case in class_struct.patient.Cases:
            rois_in_case = []
            for roi in case.PatientModel.RegionsOfInterest:
                rois_in_case.append(roi.Name)
            for roi, color in zip(['Retro_GTV', 'Retro_Ablation', 'Retro_GTV_Recurred'], ['Green', 'Teal', 'Red']):
                if roi not in rois_in_case:
                    case.PatientModel.CreateRoi(Name=roi, Color=color, Type="Organ")
            class_struct.patient.Save()
            fid = open(os.path.join(status_path, 'Created ROIs_{}.txt'.format(MRN)), 'w+')
            fid.close()


if __name__ == "__main__":
    main()
