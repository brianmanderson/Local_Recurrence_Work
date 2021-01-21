__author__ = 'Brian M Anderson'
# Created on 11/20/2020

import os
from connect import *
import time, getpass
import pandas as pd


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    df = df.loc[(pd.isnull(df['Registered'])) & (df['Has_Liver'] == 1)]
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
        if primary not in pat_dict['Primary'] or secondary not in pat_dict['Secondary']:
            pat_dict['Primary'].append(primary)
            pat_dict['Secondary'].append(secondary)
            pat_dict['Case_Number'].append(case)
        MRN_dictionary[MRN] = pat_dict
    return MRN_dictionary


def ComputeRigidRegistration(case, RefCT, AblationCT, RoiNames = None):
    perform_rigid_reg = True
    for registration in case.Registrations:
        if not registration.RegistrationSource:
            continue
        tempout = [registration.RegistrationSource.FromExamination.Name,
                   registration.RegistrationSource.ToExamination.Name]
        if RefCT in tempout and AblationCT in tempout:
            perform_rigid_reg = False

    if perform_rigid_reg:
        case.ComputeRigidImageRegistration(FloatingExaminationName=AblationCT, ReferenceExaminationName=RefCT,
                                           UseOnlyTranslations=False, HighWeightOnBones=False, InitializeImages=True,
                                           FocusRoisNames=RoiNames, RegistrationName=None)
    return None


class ChangePatient(object):
    def __init__(self):
        self.patient_db = get_current("PatientDB")
        self.MRN = 0

    def change_patient(self, MRN):
        print('got here')
        self.MRN = MRN
        info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=False)
        if not info_all:
            info_all = self.patient_db.QueryPatientInfo(Filter={"PatientID": self.MRN}, UseIndexService=True)
        for info in info_all:
            if info['PatientID'] == self.MRN and info['LastName'].find('_') == -1:
                return self.patient_db.LoadPatient(PatientInfo=info, AllowPatientUpgrade=True)
        print('did not find a patient')
        return None


def simplify_contours(case,exam_name,roi_name):
    case.PatientModel.StructureSets[exam_name].SimplifyContours(RoiNames=[roi_name], RemoveHoles3D=True,
                                                                RemoveSmallContours=True, AreaThreshold=5,
                                                                ReduceMaxNumberOfPointsInContours=False,
                                                                MaxNumberOfPoints=None, CreateCopyOfRoi=False,
                                                                ResolveOverlappingContours=True)
    return None


def is_BC_roi(case, reference_examination_name, target_examination_name, roi_name):
    """
    Checks if an ROI can be used as boundary conditions. i.e., triangle mesh with the same number of vertices in both reference and target image
    :param case: Current case
    :param reference_examination_name: Name of reference examination
    :param target_examination_name: Name of target examination
    :param roi_name: Name of ROI
    :return: if ok to use
    """
    reference_ss = case.PatientModel.StructureSets[reference_examination_name]
    target_ss = case.PatientModel.StructureSets[target_examination_name]
    ref_geometry = reference_ss.RoiGeometries[roi_name]
    tar_geometry = target_ss.RoiGeometries[roi_name]
    if ref_geometry.PrimaryShape is None:
        return False
    if tar_geometry.PrimaryShape is None:
        return False
    ref_vert = None
    try:
        ref_vert = ref_geometry.PrimaryShape.Vertices
    except:
        return False
    tar_vert = None
    try:
        tar_vert = tar_geometry.PrimaryShape.Vertices
    except:
        return False
    if len([vert for vert in ref_vert]) == 0:
        return False
    if len([vert for vert in ref_vert]) != len([vert for vert in tar_vert]):
        return False
    return True


def create_dir(patient, case, Ref, Ablation, roi_base, rois_in_case):
    for exam_name in [Ref, Ablation]:
        simplify_contours(case, exam_name, roi_base)
    tag = {'Group': (0x020), 'Element': (0x0052)}
    frame_of_ref_ref = \
        case.Examinations[Ref].GetStoredDicomTagValueForVerification(**tag)['Frame of Reference UID']
    frame_of_ref_sec = \
        case.Examinations[Ablation].GetStoredDicomTagValueForVerification(**tag)['Frame of Reference UID']
    if frame_of_ref_ref == frame_of_ref_sec:
        set_progress('Assigning secondary image to new frame of reference')
        patient.Save()
        case.Examinations[Ablation].AssignToNewFrameOfReference()
    ComputeRigidRegistration(case=case, RefCT=Ref, AblationCT=Ablation, RoiNames=[roi_base])
    use_curvature_adaptation = False
    equal_edge = 3
    smooth_iter = 1
    max_smooth_steps = 4
    voxel_side_length = 0.2
    BC_rois = [roi.Name for roi in case.PatientModel.RegionsOfInterest if roi.Name.startswith(roi_base) and
               is_BC_roi(case, Ref, Ablation, roi.Name)]
    if BC_rois:
        BC_roi_name = BC_rois[-1]
    else:
        try:
            smoothing = case.PatientModel.CreateBoundaryConditionsForMorfeus(
                RoiNames=[roi_base],
                ReferenceExaminationName=Ref, TargetExaminationName=Ablation,
                VoxelRepresentationVoxelSideLength=voxel_side_length,
                TriangulationEqualEdgeLength=equal_edge,
                TriangulationUseCurvatureAdaptation=use_curvature_adaptation,
                TriangulationSmoothingIterations=smooth_iter,
                MaxNrOfSmoothingSteps=max_smooth_steps)
            BC_roi_name = [roi.Name for roi in case.PatientModel.RegionsOfInterest if roi.Name not in
                           rois_in_case][0]
            print(BC_roi_name)
        except Exception as e:
            MessageBox.Show("Error when generating BCs")
            return
    BioDef_Name = 'Deform_BCs_{}_to_{}_for_{}'.format(Ref, Ablation, roi_base)[:50]  # Must be less than 50 characters
    group_names = [group.Name == BioDef_Name for group in case.PatientModel.StructureRegistrationGroups]
    if group_names:
        already_done = max(group_names)
    else:
        already_done = False
    if not already_done:
        case.PatientModel.CreateBiomechanicalDeformableRegistrationGroup(RegistrationGroupName=BioDef_Name,
                                                                         ReferenceExaminationName=Ref,
                                                                         TargetExaminationNames=[
                                                                             Ablation],
                                                                         ControllingRois=[{'Name':
                                                                                               BC_roi_name,
                                                                                           'InterfaceModeling':
                                                                                               "Fixed"}],
                                                                         DeformationGridVoxelSize={'x': 0.25,
                                                                                                   'y': 0.25,
                                                                                                   'z': 0.25})
    patient.Save()
    return None


def main():
    base_export_path = r'H:\Deeplearning_Recurrence_Work\Deformed_Exports'
    if not os.path.exists(base_export_path):
        os.makedirs(base_export_path)
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'

    MRN_dictionary = return_MRN_dictionary(excel_path)
    patient_changer = ChangePatient()
    current_MRN = None
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        for primary, secondary, case_num in zip(MRN_dictionary[MRN_key]['Primary'],
                                                MRN_dictionary[MRN_key]['Secondary'],
                                                MRN_dictionary[MRN_key]['Case_Number']):

            out_deformation_image = os.path.join(base_export_path, '{}_{}_to_{}.mhd'.format(MRN, primary, secondary))
            if os.path.exists(out_deformation_image):
                print('{} was already deformed'.format(MRN))
                continue
            patient = None
            if current_MRN != MRN:
                try:
                    patient = patient_changer.change_patient(MRN)
                    current_MRN = MRN
                except:
                    current_MRN = None
                    continue
            if patient is None:
                print('{} failed to load a patient'.format(MRN))
                continue

            print(MRN)
            case = None
            for case in patient.Cases:
                if int(case.CaseName.split(' ')[-1]) == int(case_num):
                    break
                continue

            already_deformed = False
            new_reg_name = 'Deform_BCs_{}_to_{}'.format(primary, secondary)
            old_reg_name = 'Deformation_Boundary_Conditions_{}_to_{}'.format(primary, secondary)
            older_name = 'BiomechanicalDefReg_Liver_TriMesh_{}_{}'.format(primary, secondary)
            for top_registration in case.Registrations:
                for struct_reg in top_registration.StructureRegistrations:
                    for name in [new_reg_name, older_name, old_reg_name]:
                        if struct_reg.Name.startswith(new_reg_name) or struct_reg.Name.startswith(name):
                            already_deformed = True
                            # if not os.path.exists(out_deformation_image):
                            #     struct_reg.ExportDeformedMetaImage(MetaFileName=out_deformation_image)
                            break
            if already_deformed:
                print('{} was already deformed'.format(MRN))
                continue
            rois_in_case = []
            for roi in case.PatientModel.RegionsOfInterest:
                rois_in_case.append(roi.Name)
            for roi_base in ['Liver', 'Liver_BMA_Program_4']:
                has_contours = True
                if roi_base not in rois_in_case:
                    has_contours = False
                    continue
                for exam in [primary, secondary]:
                    if not case.PatientModel.StructureSets[exam].RoiGeometries[roi_base].HasContours():
                        has_contours = False
                        break
                if has_contours:
                    break
            if not has_contours:
                print('{} did not have the contours needed'.format(MRN))
                continue
            create_dir(patient, case, primary, secondary, roi_base, rois_in_case)
            # break
            '''
            Now export the meta image
            '''
            # for top_registration in case.Registrations:
            #     for structure_registration in top_registration.StructureRegistrations:
            #         if structure_registration.Name.startswith(new_reg_name):
            #             if not os.path.exists(out_deformation_image):
            #                 structure_registration.ExportDeformedMetaImage(MetaFileName=out_deformation_image)
            #             break


if __name__ == '__main__':
    main()
