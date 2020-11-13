__author__ = 'Brian M Anderson'
# Created on 2/3/2020

from connect import *


def map_vasculature(case, examination,rois_in_case=[],is_primary=True, expansion=5):
    liver_roi = r"Liver_Ablation"
    ablation_roi = r"GTV"
    vasc_base = r"Liver_Vasculature"

    if not is_primary:
        liver_roi = r"Liver"
        ablation_roi = r"Ablation_Recurrence"

    for roi in [liver_roi, ablation_roi, vasc_base]:
        print(roi)
        assert roi in rois_in_case, 'Need to provide contours of {}!'.format(roi)
        assert case.PatientModel.StructureSets[examination.Name].RoiGeometries[roi].HasContours(), \
            '{} not defined on exam!'.format(roi)

    colours = ['Brown', 'Purple', 'Pink']
    cof_roi = r"Center_Of_Interest"
    exp_roi = r"Expanded_ROI"
    vasc_roi = r"Vasculature_Within_ROI"
    for color, new_roi in zip(colours, [cof_roi, exp_roi, vasc_roi]):
        if new_roi not in rois_in_case:
            case.PatientModel.CreateRoi(Name=new_roi, Color=color, Type="Organ")
            rois_in_case.append(new_roi)

    center = case.PatientModel.StructureSets[examination.Name].RoiGeometries[ablation_roi].GetCenterOfRoi()
    output = {'x': center.x, 'y': center.y, 'z': center.z}
    case.PatientModel.RegionsOfInterest[cof_roi].CreateSphereGeometry(Radius=0.1, Examination=examination,
                                                                      Center=output)

    case.PatientModel.RegionsOfInterest[exp_roi].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                                                       ExpressionA={'Operation': "Union",
                                                                                    'SourceRoiNames': [cof_roi],
                                                                                    'MarginSettings': {
                                                                                        'Type': "Expand",
                                                                                        'Superior': expansion,
                                                                                        'Inferior': expansion,
                                                                                        'Anterior': expansion,
                                                                                        'Posterior': expansion,
                                                                                        'Right': expansion,
                                                                                        'Left': expansion}},
                                                                       ExpressionB={'Operation': "Union",
                                                                                    'SourceRoiNames': [liver_roi],
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

    case.PatientModel.RegionsOfInterest[vasc_roi].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                                                        ExpressionA={'Operation': "Union",
                                                                                     'SourceRoiNames': [vasc_base],
                                                                                     'MarginSettings': {
                                                                                         'Type': "Expand",
                                                                                         'Superior': 0,
                                                                                         'Inferior': 0,
                                                                                         'Anterior': 0,
                                                                                         'Posterior': 0, 'Right': 0,
                                                                                         'Left': 0}},
                                                                        ExpressionB={'Operation': "Union",
                                                                                     'SourceRoiNames': [exp_roi],
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
                                                                                              'Right': 0,
                                                                                              'Left': 0})


case = get_current("Case")
patient = get_current("Patient")
primary_exam = 'CT 7'
secondary_exam = 'CT 8'
expansion = 7  # cm
rois_in_case = []
for name in case.PatientModel.RegionsOfInterest:
    print(name.Name)
    rois_in_case.append(name.Name)
for exam, is_primary in zip([primary_exam, secondary_exam],[True,False]):
    examination = case.Examinations[exam]
    map_vasculature(case,examination,rois_in_case=rois_in_case,is_primary=is_primary, expansion=expansion)
patient.Save()
case.ComputeRigidROIRegistration(FloatingExaminationName=secondary_exam, ReferenceExaminationName=primary_exam,
                                 DiscardRotations=False, RoiNames=[r"Vasculature_Within_ROI"])
map_rois = [r"Liver_Ablation", r"Ablation", r"GTV_Exp_5mm_outside_Ablation", ]
if r"GTV_Exp_7.5mm_outside_Ablation" in rois_in_case:
    map_rois.append(r"GTV_Exp_7.5mm_outside_Ablation")
case.PatientModel.CopyRoiGeometries(SourceExamination=case.Examinations[primary_exam],
                                    TargetExaminationNames=[secondary_exam],
                                    RoiNames=map_rois)
patient.Save()