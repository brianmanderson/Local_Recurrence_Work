__author__ = 'Brian M Anderson'
# Created on 4/14/2021
"""
This is code created to update the exam names in our excel sheets and folder paths
Exam names changed after the migration from General database to Brocklab database
"""
from connect import *
import os


class ChangePatient(object):
    def __init__(self):
        self.patient_db = get_current('PatientDB')

    def changepatient(self, MRN):
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


def create_exam_text_name(case, path):
    rois_in_case = []
    for roi in case.PatientModel.RegionsOfInterest:
        rois_in_case.append(roi.Name)
    for root, folders, files in os.walk(path):
        print(root)
        if 'SeriesInstanceUID.txt' in files:
            fid = open(os.path.join(root, 'SeriesInstanceUID.txt'))
            series_instance_uid = fid.readline()
            fid.close()
            for exam in case.Examinations:
                if exam.Series[0].ImportedDicomUID == series_instance_uid:
                    fid = open(os.path.join(root, 'New_Exam_{}.txt'.format(exam.Name)), 'w+')
                    fid.close()
    return None


def main():
    path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    patient_changer = ChangePatient()
    for MRN in os.listdir(path):
        try:
            patient_changer.changepatient(MRN)
        except:
            continue
        patient = patient_changer.patient
        if patient is not None:
            patient_path = os.path.join(path, MRN)
            for case in patient.Cases:
                case_path = os.path.join(patient_path, case.CaseName)
                print(case_path)
                if os.path.exists(case_path):
                    create_exam_text_name(case=case, path=case_path)
    return None


if __name__ == '__main__':
    main()
