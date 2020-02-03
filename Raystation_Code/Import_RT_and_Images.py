__author__ = 'Brian M Anderson'
# Created on 2/3/2020

from connect import *
import os


patient = get_current('Patient')
patient_db = get_current("PatientDB")
exam = get_current("Examination")
case = get_current("Case")
MRN = patient.PatientID


path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
check_path = os.path.join(path,patient.PatientID,exam.Name,'new_RT')
pi = patient_db.QueryPatientsFromPath(Path=check_path, SearchCriterias={'PatientID': MRN})[0]
studies = patient_db.QueryStudiesFromPath(Path=check_path,SearchCriterias=pi)
series = []
for study in studies:
    series += patient_db.QuerySeriesFromPath(Path=check_path,SearchCriterias=study)
patient.ImportDataFromPath(Path=check_path, CaseName=case.CaseName,SeriesOrInstances=series,
                           AllowMismatchingPatientID=True)
for file in os.listdir(check_path):
    os.remove(os.path.join(check_path,file))