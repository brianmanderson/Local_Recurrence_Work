__author__ = 'Brian M Anderson'
# Created on 2/3/2020

from connect import *

case = get_current("Case")
primary = 'CT 8'
secondary = 'CT 1'
case.PatientModel.CopyRoiGeometries(SourceExamination=case.Examinations[primary], TargetExaminationNames=[secondary],
                                    RoiNames=[r"Liver_Ablation", r"Ablation", r"GTV_Exp_5mm_outside_Ablation"])
