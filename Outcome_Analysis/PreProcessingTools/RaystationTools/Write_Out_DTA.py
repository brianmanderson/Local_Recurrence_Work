__author__ = 'Brian M Anderson'
# Created on 11/23/2020
from connect import *
import os
import pandas as pd


def actual_Map(case,Ref,Ablation,BioDef_Name,base_liver_name='Liver_BMA_Program_4',gtv_name='GTV',ablation_name='Ablation',
               rois_in_case=[]):
    CreateRoi = False
    case.MapRoiGeometriesDeformably(RoiGeometryNames=[gtv_name], CreateNewRois=CreateRoi,
                                    StructureRegistrationGroupNames=[BioDef_Name],
                                    ReferenceExaminationNames=[Ref], TargetExaminationNames=[Ablation],
                                    ReverseMapping=False, AbortWhenBadDisplacementField=False)
    case.MapRoiGeometriesDeformably(RoiGeometryNames=[ablation_name], CreateNewRois=CreateRoi,
                                    StructureRegistrationGroupNames=[BioDef_Name],
                                    ReferenceExaminationNames=[Ref], TargetExaminationNames=[Ablation],
                                    ReverseMapping=True, AbortWhenBadDisplacementField=False)
    return None


def MapRoiGeometries(case, Ref, Ablation, BioDef_Name = 'BiomechanicalDefReg_Liver_TriMesh',base_liver_name='Liver',
                     gtv_name='GTV',ablation_name='Ablation', rois_in_case=[]):
    BioDef_Name_Base = BioDef_Name
    for z in range(0,9):
        try:
            actual_Map(case, Ref, Ablation, BioDef_Name, base_liver_name=base_liver_name,
                       gtv_name=gtv_name,ablation_name=ablation_name, rois_in_case=rois_in_case)
            break
        except:
            BioDef_Name = BioDef_Name_Base + str(z)
            continue
    return None


def volume_expansion(roi_base,gtv_name,ablation_roi,rois_in_case,case, examination):
    examination = case.Examinations[examination]
    '''
    This is for doing min distance measurements
    '''
    exp = 3
    gtv_exp_name = '{}_Expansion_{}cm_within_liver'.format(gtv_name,exp)
    print(rois_in_case)
    if gtv_exp_name not in rois_in_case:
        case.PatientModel.CreateRoi(Name=gtv_exp_name, Color="Red", Type="Organ", TissueName=None,
                                    RbeCellTypeName=None, RoiMaterial=None)
        rois_in_case.append(gtv_exp_name)
    case.PatientModel.RegionsOfInterest[gtv_exp_name].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                                                       ExpressionA={'Operation': "Union",
                                                                                    'SourceRoiNames': [gtv_name],
                                                                                    'MarginSettings': {
                                                                                        'Type': "Expand",
                                                                                        'Superior': exp,
                                                                                        'Inferior': exp,
                                                                                        'Anterior': exp,
                                                                                        'Posterior': exp,
                                                                                        'Right': exp, 'Left': exp}},
                                                                       ExpressionB={'Operation': "Union",
                                                                                    'SourceRoiNames': [roi_base],
                                                                                    'MarginSettings': {
                                                                                        'Type': "Expand",
                                                                                        'Superior': 0,
                                                                                        'Inferior': 0,
                                                                                        'Anterior': 0,
                                                                                        'Posterior': 0, 'Right': 0,
                                                                                        'Left': 0}},
                                                                       ResultOperation="Intersection",
                                                                       ResultMarginSettings={'Type': "Expand",
                                                                                             'Superior': 0,
                                                                                             'Inferior': 0,
                                                                                             'Anterior': 0,
                                                                                             'Posterior': 0,
                                                                                             'Right': 0, 'Left': 0})

    exp = 5
    gtv_exp_name = '{}_Expansion_{}cm_within_liver'.format(gtv_name,exp)
    print(rois_in_case)
    if gtv_exp_name not in rois_in_case:
        case.PatientModel.CreateRoi(Name=gtv_exp_name, Color="Red", Type="Organ", TissueName=None,
                                    RbeCellTypeName=None, RoiMaterial=None)
        rois_in_case.append(gtv_exp_name)
    case.PatientModel.RegionsOfInterest[gtv_exp_name].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                                                       ExpressionA={'Operation': "Union",
                                                                                    'SourceRoiNames': [gtv_name],
                                                                                    'MarginSettings': {
                                                                                        'Type': "Expand",
                                                                                        'Superior': exp,
                                                                                        'Inferior': exp,
                                                                                        'Anterior': exp,
                                                                                        'Posterior': exp,
                                                                                        'Right': exp, 'Left': exp}},
                                                                       ExpressionB={'Operation': "Union",
                                                                                    'SourceRoiNames': [roi_base],
                                                                                    'MarginSettings': {
                                                                                        'Type': "Expand",
                                                                                        'Superior': 0,
                                                                                        'Inferior': 0,
                                                                                        'Anterior': 0,
                                                                                        'Posterior': 0, 'Right': 0,
                                                                                        'Left': 0}},
                                                                       ResultOperation="Intersection",
                                                                       ResultMarginSettings={'Type': "Expand",
                                                                                             'Superior': 0,
                                                                                             'Inferior': 0,
                                                                                             'Anterior': 0,
                                                                                             'Posterior': 0,
                                                                                             'Right': 0, 'Left': 0})
    gtv_exp_hollow = '{}_Expand_Hollow'.format(gtv_name)
    if gtv_exp_hollow not in rois_in_case:
        rois_in_case.append(gtv_exp_hollow)
        case.PatientModel.CreateRoi(Name=gtv_exp_hollow, Color="Red", Type="Organ", TissueName=None,
                                    RbeCellTypeName=None, RoiMaterial=None)
    print('got here')
    case.PatientModel.RegionsOfInterest[gtv_exp_hollow].CreateAlgebraGeometry(Examination=examination,
                                                                                        Algorithm="Auto",
                                                                                        ExpressionA={
                                                                                            'Operation': "Union",
                                                                                            'SourceRoiNames': [
                                                                                                gtv_exp_name],
                                                                                            'MarginSettings': {
                                                                                                'Type': "Expand",
                                                                                                'Superior': 0.0,
                                                                                                'Inferior': 0.0,
                                                                                                'Anterior': 0.0,
                                                                                                'Posterior': 0.0,
                                                                                                'Right': 0.0,
                                                                                                'Left': 0.0}},
                                                                                        ExpressionB={
                                                                                            'Operation': "Union",
                                                                                            'SourceRoiNames': [
                                                                                                ablation_roi],
                                                                                            'MarginSettings': {
                                                                                                'Type': "Expand",
                                                                                                'Superior': 0,
                                                                                                'Inferior': 0,
                                                                                                'Anterior': 0,
                                                                                                'Posterior': 0,
                                                                                                'Right': 0,
                                                                                                'Left': 0}},
                                                                                        ResultOperation="Subtraction",
                                                                                        ResultMarginSettings={
                                                                                            'Type': "Expand",
                                                                                            'Superior': 0,
                                                                                            'Inferior': 0,
                                                                                            'Anterior': 0,
                                                                                            'Posterior': 0,
                                                                                            'Right': 0, 'Left': 0})
    case.PatientModel.RegionsOfInterest[gtv_exp_name].DeleteRoi()
    del rois_in_case[rois_in_case.index(gtv_exp_name)]
    '''
    This is for doing expansions outside of the ablated zone
    '''
    exps = [.5,.75]
    descs = ['_Exp_5mm','_Exp_7.5mm']
    for exp,desc in zip(exps,descs):
        gtv_exp = gtv_name + desc
        if gtv_exp not in rois_in_case:
            rois_in_case.append(gtv_exp)
            case.PatientModel.CreateRoi(Name=gtv_exp, Color="Red", Type="Organ", TissueName=None,
                                                   RbeCellTypeName=None, RoiMaterial=None)

        case.PatientModel.RegionsOfInterest[gtv_exp].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                       ExpressionA={'Operation': "Union", 'SourceRoiNames': [gtv_name],
                                                    'MarginSettings': {'Type': "Expand", 'Superior': exp,
                                                                       'Inferior': exp, 'Anterior': exp,
                                                                       'Posterior': exp, 'Right': exp, 'Left': exp}},
                                       ExpressionB={'Operation': "Union", 'SourceRoiNames': [roi_base],
                                                    'MarginSettings': {'Type': "Expand", 'Superior': 0, 'Inferior': 0,
                                                                       'Anterior': 0, 'Posterior': 0, 'Right': 0,
                                                                       'Left': 0}}, ResultOperation="Intersection",
                                       ResultMarginSettings={'Type': "Expand", 'Superior': 0, 'Inferior': 0,
                                                             'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0})

        gtv_exp_ablation = gtv_exp + '_Ablation'
        if gtv_exp_ablation not in rois_in_case:
            rois_in_case.append(gtv_exp_ablation)
            case.PatientModel.CreateRoi(Name=gtv_exp_ablation, Color="Red", Type="Organ", TissueName=None,
                                        RbeCellTypeName=None, RoiMaterial=None)

        case.PatientModel.RegionsOfInterest[gtv_exp_ablation].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                       ExpressionA={'Operation': "Union", 'SourceRoiNames': [gtv_exp],
                                                    'MarginSettings': {'Type': "Expand", 'Superior': 0.0,
                                                                       'Inferior': 0.0, 'Anterior': 0.0,
                                                                       'Posterior': 0.0, 'Right': 0.0,
                                                                       'Left': 0.0}},
                                       ExpressionB={'Operation': "Union",
                                                    'SourceRoiNames': [ablation_roi],
                                                    'MarginSettings': {'Type': "Expand", 'Superior': 0,
                                                                       'Inferior': 0, 'Anterior': 0, 'Posterior': 0,
                                                                       'Right': 0, 'Left': 0}},
                                       ResultOperation="Intersection",
                                       ResultMarginSettings={'Type': "Expand", 'Superior': 0, 'Inferior': 0,
                                                             'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0})
        gtv_exp_outside_ablation = gtv_exp + '_outside_Ablation'
        if gtv_exp_outside_ablation not in rois_in_case:
            rois_in_case.append(gtv_exp_outside_ablation)
            case.PatientModel.CreateRoi(Name=gtv_exp_outside_ablation, Color="Red", Type="Organ", TissueName=None,
                                                   RbeCellTypeName=None, RoiMaterial=None)

        case.PatientModel.RegionsOfInterest[gtv_exp_outside_ablation].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                       ExpressionA={'Operation': "Union", 'SourceRoiNames': [gtv_exp],
                                                    'MarginSettings': {'Type': "Expand", 'Superior': 0.0,
                                                                       'Inferior': 0.0, 'Anterior': 0.0,
                                                                       'Posterior': 0.0, 'Right': 0.0,
                                                                       'Left': 0.0}},
                                       ExpressionB={'Operation': "Union",
                                                    'SourceRoiNames': [ablation_roi],
                                                    'MarginSettings': {'Type': "Expand", 'Superior': 0,
                                                                       'Inferior': 0, 'Anterior': 0, 'Posterior': 0,
                                                                       'Right': 0, 'Left': 0}},
                                       ResultOperation="Subtraction",
                                       ResultMarginSettings={'Type': "Expand", 'Superior': 0, 'Inferior': 0,
                                                             'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0})
    for exp,desc in zip(exps,descs):
        gtv_exp = gtv_name + desc
        gtv_exp_ablation = gtv_exp + '_Ablation'
        print(gtv_exp)
        data = case.PatientModel.StructureSets[examination.Name].ComparisonOfRoiGeometries(RoiA=gtv_exp,RoiB=gtv_exp_ablation)
        for i in data:
            print(i)
        gtv_exp_volume = case.PatientModel.StructureSets[examination.Name].RoiGeometries[gtv_exp].GetRoiVolume()
        gtv_exp_volume_ablation = case.PatientModel.StructureSets[examination.Name].RoiGeometries[gtv_exp_ablation].GetRoiVolume()
        print(str(gtv_exp_volume/gtv_exp_volume_ablation*100) + '% covered')
        case.PatientModel.RegionsOfInterest[gtv_exp].DeleteRoi()
        del rois_in_case[rois_in_case.index(gtv_exp)]
        case.PatientModel.RegionsOfInterest[gtv_exp_ablation].DeleteRoi()
        del rois_in_case[rois_in_case.index(gtv_exp_ablation)]
    return rois_in_case


def GetVolume_and_DTA(case, Ref, gtv_name='GTV',ablation_name='Ablation'):
    volExcluded = {}
    volExcluded['GTV_volume'] = case.PatientModel.StructureSets[Ref].RoiGeometries[gtv_name].GetRoiVolume()
    volExcluded['Ablation_volume'] = case.PatientModel.StructureSets[Ref].RoiGeometries[ablation_name].GetRoiVolume()

    dict_val = case.PatientModel.StructureSets[Ref].RoiSurfaceToSurfaceDistanceBasedOnDT(ReferenceRoiName=gtv_name,TargetRoiName=gtv_name + "_Expand_Hollow")
    case.PatientModel.RegionsOfInterest[gtv_name + "_Expand_Hollow"].DeleteRoi()
    min_val = dict_val['Min']
    volExcluded['Min DTA'] = min_val
    volExcluded['Aver DTA'] = dict_val['Average']
    return volExcluded


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='DTA')
    MRN_list, case_list, primary_list, secondary_list = df['MRN'].values, df['Case'].values, df['Primary'].values,\
                                                        df['Secondary'].values
    MRN_dictionary = {}
    for MRN, case, primary, secondary in zip(MRN_list, case_list, primary_list, secondary_list):
        if str(df['Rigid_DTA'].values[list(df['MRN'].values).index(MRN)]) != 'nan':
            continue
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
        MRN_dictionary[MRN] = {'Primary': primary, 'Secondary': secondary, 'Case': str(case)}
    return MRN_dictionary, df


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


def get_contour_points_from_roi(geometry):
    """
    :param geometry: Geometry from case.PatientModel.StructureSets[CT x].RoiGeometries['Roi_name']
    :return:
    """
    try:
        old_contours = geometry.PrimaryShape.Contours

        new_contours = []

        for contour in old_contours:
            new_contour = []
            for point in contour:
                new_contour.append({'x': point.x, 'y': point.y, 'z': point.z})
            new_contours.append(new_contour)

        return new_contours
    except:
        return None


def main():
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\DTA_Results_Recurrence_and_Non_Recurrence.xlsx'
    MRN_dictionary, df = return_MRN_dictionary(excel_path)
    df.set_index('MRN', inplace=True)
    patient_changer = ChangePatient()
    gtv_base = 'GTV'
    ablation_base = 'Ablation'
    for MRN_key in MRN_dictionary.keys():
        patient_dict = {'MRN': [MRN_key]}
        case, primary_exam, secondary_exam = MRN_dictionary[MRN_key]['Case'], MRN_dictionary[MRN_key]['Primary'],\
                                             MRN_dictionary[MRN_key]['Secondary']
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        try:
            patient = patient_changer.change_patient(MRN)
        except:
            continue
        if patient is None:
            print('{} failed to load a patient'.format(MRN))
            continue
        print(MRN)
        if case == '-1':
            for case in patient.Cases:
                continue
        else:
            case = patient.Cases['CASE {}'.format(case)]
        case.SetCurrent()
        rois_in_case = []
        for roi in case.PatientModel.RegionsOfInterest:
            rois_in_case.append(roi.Name)
        roi_base = None
        for roi in ['Liver', 'Liver_BMA_Program_4']:
            if roi not in rois_in_case:
                continue
            elif case.PatientModel.StructureSets[primary_exam].RoiGeometries[roi].HasContours():
                if case.PatientModel.StructureSets[secondary_exam].RoiGeometries[roi].HasContours():
                    roi_base = roi
                    break
        assert roi_base is not None, "No liver contour found"
        for roi in [roi_base, gtv_base, ablation_base]:
            assert roi in rois_in_case, '{} is not present in the case!'.format(roi)

        for roi in [roi_base, gtv_base]:
            assert case.PatientModel.StructureSets[primary_exam].RoiGeometries[roi].HasContours(), '{} is not ' \
                                                                                                   'on primary!' \
                                                                                                   ''.format(roi)
        for roi in [roi_base, ablation_base]:
            assert case.PatientModel.StructureSets[secondary_exam].RoiGeometries[roi].HasContours(), '{} is not ' \
                                                                                                   'on secondary!' \
                                                                                                   ''.format(roi)
        gtv_roi = gtv_base + '_Rigid'
        if gtv_roi not in rois_in_case:
            case.PatientModel.CreateRoi(Name=gtv_roi, Color='Red', Type='External')
        primary_examination = case.Examinations[primary_exam]
        case.PatientModel.RegionsOfInterest[gtv_roi].CreateExternalGeometry(Examination=primary_examination,
                                                                            ThresholdLevel=-250)
        contours = get_contour_points_from_roi(case.PatientModel.StructureSets[primary_exam].RoiGeometries[gtv_base])
        case.PatientModel.StructureSets[primary_exam].RoiGeometries[gtv_roi].PrimaryShape.Contours = contours
        case.PatientModel.RegionsOfInterest[gtv_roi].Type = case.PatientModel.RegionsOfInterest[gtv_base].Type
        case.PatientModel.CopyRoiGeometries(SourceExamination=case.Examinations[primary_exam],
                                            TargetExaminationNames=[secondary_exam],
                                            RoiNames=[gtv_roi])
        # save_obj(self.ExcludedVolume, self.out_path_name)

        volume_expansion(roi_base, gtv_roi, ablation_base, rois_in_case, case, secondary_exam)
        out_dta = GetVolume_and_DTA(case, secondary_exam, gtv_name=gtv_roi, ablation_name=ablation_base)
        print(out_dta)
        patient_dict['Rigid_DTA'] = [out_dta['Min DTA']]
        patient_df = pd.DataFrame(patient_dict)
        print(patient_dict)
        df.update(patient_df.set_index('MRN'))
        df.to_excel(excel_path, index=0)
        break


if __name__ == '__main__':
    main()
