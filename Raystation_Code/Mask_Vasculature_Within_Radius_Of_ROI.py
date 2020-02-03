__author__ = 'Brian M Anderson'
# Created on 2/3/2020

from connect import *

case = get_current("Case")
examination = get_current("Examination")

liver_roi = r"Liver_Ablation"
ablation_roi = r"Ablation"
vasc_base = r"Liver_Vasculature"

expansion = 5  # cm

rois_in_case = []
for name in case.PatientModel.RegionsOfInterest:
    print(name.Name)
    rois_in_case.append(name.Name)
for roi in [liver_roi,ablation_roi,vasc_base]:
    assert roi in rois_in_case, 'Need to provide {} contours!'.format(roi)
    assert case.PatientModel.StructureSets[examination.Name].RoiGeometries[roi].HasContours(), \
        '{} contours not defined on exam!'.format(roi)

cof_roi = r"Center_Of_Interest"
exp_roi = r"Expanded_ROI"
vasc_roi = r"Vasculature_Within_ROI"


colours = ['Brown','Purple','Pink']
for color, new_roi in zip(colours,[cof_roi,exp_roi,vasc_roi]):
    if new_roi not in rois_in_case:
        case.PatientModel.CreateRoi(Name=new_roi, Color=color, Type="Organ")
        rois_in_case.append(new_roi)

if not case.PatientModel.StructureSets[examination.Name].RoiGeometries[cof_roi].HasContours():
    center = case.PatientModel.StructureSets[examination.Name].RoiGeometries[ablation_roi].GetCenterOfRoi()
    output = {'x':center.x,'y':center.y,'z':center.z}
    case.PatientModel.RegionsOfInterest[cof_roi].CreateSphereGeometry(Radius=0.1,Examination=examination, Center=output)

if not case.PatientModel.StructureSets[examination.Name].RoiGeometries[exp_roi].HasContours():
    case.PatientModel.RegionsOfInterest[exp_roi].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                   ExpressionA={'Operation': "Union", 'SourceRoiNames': [cof_roi],
                                                'MarginSettings': {'Type': "Expand", 'Superior': expansion, 'Inferior': expansion,
                                                                   'Anterior': expansion, 'Posterior': expansion, 'Right': expansion,
                                                                   'Left': expansion}},
                                   ExpressionB={'Operation': "Union", 'SourceRoiNames': [liver_roi],
                                                'MarginSettings': {'Type': "Expand", 'Superior': 0, 'Inferior': 0,
                                                                   'Anterior': 0, 'Posterior': 0, 'Right': 0,
                                                                   'Left': 0}}, ResultOperation="Intersection",
                                   ResultMarginSettings={'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0,
                                                         'Posterior': 0, 'Right': 0, 'Left': 0})

if not case.PatientModel.StructureSets[examination.Name].RoiGeometries[vasc_roi].HasContours():
    case.PatientModel.RegionsOfInterest[vasc_roi].CreateAlgebraGeometry(Examination=examination, Algorithm="Auto",
                                                                        ExpressionA={ 'Operation': "Union", 'SourceRoiNames': [vasc_base], 'MarginSettings': { 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 } },
                                                                        ExpressionB={ 'Operation': "Union", 'SourceRoiNames': [exp_roi], 'MarginSettings': { 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 } },
                                                                        ResultOperation="Intersection",
                                                                        ResultMarginSettings={ 'Type': "Expand", 'Superior': 0, 'Inferior': 0, 'Anterior': 0, 'Posterior': 0, 'Right': 0, 'Left': 0 })