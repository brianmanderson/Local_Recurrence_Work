__author__ = 'Brian M Anderson'
# Created on 9/28/2020


import os
from connect import *
import time, getpass
import pandas as pd


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    MRN_list, GTV_List, Ablation_list = df['MRN'].values, df['PreExam'].values, df['Ablation_Exam'].values
    MRN_dictionary = {}
    for MRN, GTV, Ablation in zip(MRN_list, GTV_List, Ablation_list):
        if MRN not in MRN_dictionary:
            MRN_dictionary[MRN] = []
        for exam in [GTV, Ablation]:
            if type(exam) is not float:
                exam = str(exam)
                if exam.startswith('CT'):
                    if exam.find(' ') == -1:
                        exam = 'CT {}'.format(exam.split('CT')[-1])
                    if exam not in MRN_dictionary[MRN]:
                        MRN_dictionary[MRN].append(exam)
    return MRN_dictionary


class create_RT_Structure():
    def __init__(self,roi_name):
        self.roi_name = roi_name
        self.version_name = '_BMA_Program_4'
        try:
            self.patient_db = get_current('PatientDB')
            self.patient = get_current('Patient')
            self.case = get_current('Case')
            self.exam = get_current('Examination')
            self.MRN = self.patient.PatientID
        except:
            xxx = 1

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

    def create_RT_Liver(self, exam):
        self.export(exam)
        if self.has_contours:
            print('Already has contours')
            return None
        self.import_data(exam)

    def get_rois_in_case(self):
        self.rois_in_case = []
        for roi in self.case.PatientModel.RegionsOfInterest:
            self.rois_in_case.append(roi.Name)

    def check_has_contours(self, exam):
        roi_name = self.roi_name
        set_progress('Checking to see if {} already has contours'.format(self.roi_name + self.version_name))
        roi_name += '_Auto_Contour'
        self.MRN = self.patient.PatientID
        self.base_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Clinical\\Auto_Contour_Sites\\'
        try:
            os.listdir(self.base_path)
        except:
            self.base_path = '\\\\mymdafiles\\ou-radonc\\Raystation\\Research\\Auto_Contour_Sites\\'
        #if not check_any_contours(case,exam): Doesn't work the way I want it to
        self.path = os.path.join(self.base_path, self.roi_name + '_Auto_Contour', 'Input_3', self.MRN)
        self.get_rois_in_case()
        self.patient.Save()
        self.has_contours = False
        if self.roi_name + self.version_name in self.rois_in_case:
            if self.case.PatientModel.StructureSets[exam.Name].RoiGeometries[self.roi_name + self.version_name].HasContours():
                self.patient.Save()
                self.get_rois_in_case()
                self.has_contours = True
                return True # Already have the contours for this patient
        return False

    def export(self, exam):
        if not self.check_has_contours(exam):
            self.patient.Save()
            print(self.path)
            self.Export_Dicom(exam, self.path)

    def import_data(self, exam):
        roi_name = self.roi_name
        actual_roi_name = roi_name + self.version_name
        roi_name += '_Auto_Contour'
        self.get_rois_in_case()
        if actual_roi_name in self.rois_in_case:
            if self.case.PatientModel.StructureSets[exam.Name].RoiGeometries[actual_roi_name].HasContours():
                return None # Already have the contours for this patient
        data = exam.GetAcquisitionDataFromDicom()
        SeriesUID = data['SeriesModule']['SeriesInstanceUID']
        output_path = os.path.join(self.base_path,roi_name,'Output',self.MRN,SeriesUID)
        # self.cleanout_folder(output_path)
        print('Now waiting for RS to be made')
        self.import_RT = False
        self.check_folder(output_path)
        print('Import RT structure!')
        if self.import_RT:
            self.importRT(output_path)
        # self.cleanout_folder(output_path)
        return None

    def Export_Dicom(self,exam, path):
        data = exam.GetAcquisitionDataFromDicom()
        SeriesUID = data['SeriesModule']['SeriesInstanceUID']
        export_path = os.path.join(path,SeriesUID)
        if not os.path.exists(export_path):
            print('making path')
            os.makedirs(export_path)
        set_progress('Exporting dicom series')
        if not os.path.exists(os.path.join(export_path,'Completed.txt')):
            self.case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[exam.Name],
                                            RtStructureSetsForExaminations=[])
            fid = open(os.path.join(export_path,'Completed.txt'),'w+')
            fid.close()
        set_progress('Finished exporting, waiting in queue')
        return None

    def update_progress(self, output_path):
        files = [i for i in os.listdir(output_path) if i.startswith('Status')]
        for file in files:
            set_progress('{}'.format(file.split('Status_')[-1].split('.txt')[0]))

    def check_folder(self,output_path):
        print(output_path)
        while not os.path.exists(output_path):
            time.sleep(1)
        print('path exists, waiting for file')
        while not os.path.exists(os.path.join(output_path,'Completed.txt')) and not os.path.exists(os.path.join(output_path,'Failed.txt')):
            time.sleep(1)
            self.update_progress(output_path)
        if os.path.exists(os.path.join(output_path,'Completed.txt')):
            self.import_RT = True
            set_progress('Importing RT Structures')
        return None

    def importRT(self,file_path):
        try:
            self.patient.ImportDicomDataFromPath(Path=file_path,CaseName=self.case.CaseName,SeriesFilter={},ImportFilters=[])
        except:
            pi = self.patient_db.QueryPatientsFromPath(Path=file_path, SearchCriterias={'PatientID': self.MRN})[0]
            studies = self.patient_db.QueryStudiesFromPath(Path=file_path,
                                                           SearchCriterias=pi)
            series = []
            for study in studies:
                series += self.patient_db.QuerySeriesFromPath(Path=file_path,
                                                              SearchCriterias=study)
            self.patient.ImportDataFromPath(Path=file_path, CaseName=self.case.CaseName,
                                            SeriesOrInstances=series, AllowMismatchingPatientID=True)
        return None

    def cleanout_folder(self,dicom_dir):
        print('Cleaning up: Removing imported DICOMs, please check output folder for result')
        if os.path.exists(dicom_dir):
            files = [i for i in os.listdir(dicom_dir) if not i.startswith('user_')]
            for file in files:
                os.remove(os.path.join(dicom_dir,file))
            un = getpass.getuser()
            fid = open(os.path.join(dicom_dir,'user_{}.txt'.format(un)),'w+')
            fid.close()
        return None


def main():
    status_path = r'H:\Deeplearning_Recurrence_Work\Status\Liver'
    path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
           r'\Predicting_Recurrence'

    excel_path = os.path.join(path, 'RetroAblation.xlsx')
    MRN_dictionary = return_MRN_dictionary(excel_path)
    class_struct = create_RT_Structure(roi_name='Liver')
    for export in [True, False]:
        for MRN_key in MRN_dictionary.keys():
            MRN = str(MRN_key)
            while MRN[0] == '0':  # Drop the 0 from the front
                MRN = MRN[1:]
            print(MRN)
            if not MRN_dictionary[MRN_key]:
                continue
            if export and os.path.exists(os.path.join(status_path, 'Exported_Images_{}.txt'.format(MRN))):
                continue
            elif not export and os.path.exists(os.path.join(status_path, 'Imported_Predictions_{}.txt'.format(MRN))):
                continue
            try:
                class_struct.ChangePatient(MRN)
            except:
                continue
            if class_struct.patient is None:
                continue
            for case in class_struct.patient.Cases:
                class_struct.case = case
                for case in class_struct.patient.Cases:
                    class_struct.case = case
                    for exam_name in MRN_dictionary[MRN_key]:
                        try:
                            exam = case.Examinations[exam_name]
                        except:
                            break
                        if export:
                            if class_struct.export(exam):
                                fid = open(os.path.join(status_path, 'Exported_Images_{}.txt'.format(MRN)), 'w+')
                                fid.close()
                        else:
                            if not class_struct.check_has_contours(exam):
                                class_struct.import_data(exam)
                                class_struct.patient.Save()
                                fid = open(os.path.join(status_path, 'Imported_Predictions_{}.txt'.format(MRN)), 'w+')
                                fid.close()


if __name__ == "__main__":
    main()
