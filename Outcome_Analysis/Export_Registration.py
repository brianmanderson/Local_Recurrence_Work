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
    MRN_list, GTV_List, Ablation_list, Registered_list = df['MRN'].values, df['PreExam'].values,\
                                                         df['Ablation_Exam'].values, df['Registered'].values
    MRN_dictionary = {}
    for MRN, GTV, Ablation, Registered in zip(MRN_list, GTV_List, Ablation_list, Registered_list):
        Registered = str(Registered)
        if Registered != '1.0':
            continue
        add = True
        if type(GTV) is float or type(Ablation) is float:
            add = False
        if add:
            GTV = str(GTV)
            if GTV.startswith('CT'):
                if GTV.find(' ') == -1:
                    GTV = 'CT {}'.format(GTV.split('CT')[-1])
            Ablation = str(Ablation)
            if Ablation.startswith('CT'):
                if Ablation.find(' ') == -1:
                    Ablation = 'CT {}'.format(Ablation.split('CT')[-1])
            MRN_dictionary[MRN] = {'Primary': GTV, 'Secondary': Ablation}
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
        patient_dictionary = MRN_dictionary[MRN_key]
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
        export_path = os.path.join(base_export_path, MRN, case.CaseName, 'Registration')
        if os.path.exists(export_path) and os.listdir(export_path):
            dicom_files = [i for i in os.listdir(export_path) if i.endswith('.dcm') or
                           i.startswith('SameFrameOfReference')]
            if dicom_files:
                print('Already has a registration')
                continue  # Path already exists and has files
        primary = patient_dictionary['Primary']
        secondary = patient_dictionary['Secondary']
        had_reg = False
        for registration in case.Registrations:
            had_reg = True
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
        if not had_reg:
            print('{} did not have  registration'.format(MRN))

if __name__ == "__main__":
    main()
