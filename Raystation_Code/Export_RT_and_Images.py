__author__ = 'Brian M Anderson'
# Created on 12/5/2019

from connect import *
import os

case = get_current("Case")
exam = get_current("Examination")
patient = get_current("Patient")
patient.Save()
path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
export_path = os.path.join(path,patient.PatientID,exam.Name)
if not os.path.exists(export_path):
    os.makedirs(export_path)
    case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[exam.Name],
                               RtStructureSetsForExaminations=[exam.Name])
else:
    RS_Files = [i for i in os.listdir(export_path) if i.find('RS') == 0]
    for file in RS_Files:
        os.remove(os.path.join(export_path,file))
    case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[],
                               RtStructureSetsForExaminations=[exam.Name])