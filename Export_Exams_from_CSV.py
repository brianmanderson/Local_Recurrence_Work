__author__ = 'Brian M Anderson'
# Created on 7/1/2020

__author__ = 'Brian M Anderson'
# Created on 3/23/2020

from connect import *
import os
clr.AddReference('System.Windows.Forms')
from System.Windows.Forms import OpenFileDialog, DialogResult

def ChangePatient(patient_db, MRN):
    info_all = patient_db.QueryPatientInfo(Filter={"PatientID": MRN})
    # If it isn't, see if it's in the secondary database
    if not info_all:
        info_all = patient_db.QueryPatientInfo(Filter={"PatientID": MRN}, UseIndexService=True)
    info = []
    for info_temp in info_all:
        last_name = info_temp['LastName'].split('_')[-1]
        is_copy = False
        if last_name:
            for i in range(5):
                if last_name == str(i):
                    is_copy = True
                    break
        if info_temp['PatientID'] == MRN and not is_copy:
            info = info_temp
            break
    # If it isn't, see if it's in the secondary database
    patient = patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
    return patient

def ChangePatient_UPMC(patient_db, MRN):
    info_all = patient_db.QueryPatientInfo(Filter={"FirstName": MRN,"LastName":'PT'})
    # If it isn't, see if it's in the secondary database
    if not info_all:
        info_all = patient_db.QueryPatientInfo(Filter={"PatientID": MRN}, UseIndexService=True)
    info = []
    for info_temp in info_all:
        if info_temp['FirstName'] == MRN:
            info = info_temp
            break
    # If it isn't, see if it's in the secondary database
    patient = patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
    return patient

def Export_Dicom(path,case,exam):
    if not os.path.exists(path):
        print('making path')
        os.makedirs(path)
    if not os.path.exists(os.path.join(path,'Completed.txt')): # If it has been previously uploaded, don't do it again
        case.ScriptableDicomExport(ExportFolderPath=path,Examinations=[], RtStructureSetsForExaminations=[exam.Name])
        case.ScriptableDicomExport(ExportFolderPath=path, Examinations=[exam.Name],
                                        RtStructureSetsForExaminations=[])
    return None

def get_file():
    dialog = OpenFileDialog()
    dialog.Filter = "All Files|*.*"
    result = dialog.ShowDialog()
    Path_File = ''
    if result == DialogResult.OK:
        Path_File = dialog.FileName
    if Path_File:
        return Path_File
    else:
        return None


def main(path):
    patient_db = get_current('PatientDB')
    file = get_file()
    if file:
        fid = open(file)
    else:
        print('File not selected')
        return None
    data = {}
    fid.readline() # Get rid of headers
    for line in fid:
        line = line.strip('\n')
        line = line.split(',')
        data[line[0]] = line[1:3]
    fid.close()
    if not os.path.exists(path):
        print('Path:' + path + ' does not exist')
        return None
    wanted_rois = ['Ablation', 'Recurrence']
    for MRN in data:
        print(MRN)
        try:
            patient = ChangePatient(patient_db,MRN)
        except:
            continue
        '''
        Get all of the exams from the cases
        '''
        for case in patient.Cases:
            rois_in_case = []
            for roi in case.PatientModel.RegionsOfInterest:
                rois_in_case.append(roi.Name)
            continue_on = False
            for roi in wanted_rois:
                if roi not in rois_in_case:
                    continue_on = True
            if continue_on:
                continue
            for exam in data[MRN]:
                out_path_export = os.path.join(path, MRN, case.CaseName, exam) # case.CaseName,
                skip = True
                for roi in wanted_rois:
                    if case.PatientModel.StructureSets[exam].RoiGeometries[roi].HasContours():
                        skip = False
                if not skip:
                    if not os.path.exists(out_path_export):
                        os.makedirs(out_path_export)
                        case.ScriptableDicomExport(ExportFolderPath=out_path_export, Examinations=[exam],
                                                   RtStructureSetsForExaminations=[exam])


if __name__ == '__main__':
    out_path = r'H:\Data\Local_Recurrence_Exports'
    main(out_path)
