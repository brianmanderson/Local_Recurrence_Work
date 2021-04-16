__author__ = 'Brian M Anderson'
# Created on 4/9/2021

from connect import *
import os


export_path = r'H:\CreatingQualitativeMaps\Pat'
out_deformation_image = os.path.join(export_path, 'Deformed.mhd')
patient = get_current("Patient")
case = get_current("Case")
exam = get_current("Examination")
primary = 'CT 1'
secondary = 'CT 2'
case.ScriptableDicomExport(ExportFolderPath=export_path, Examinations=[primary],
                           RtStructureSetsForExaminations=[primary])
new_reg_name = 'Deform_BCs_{}_to_{}'.format(primary, secondary)
for top_registration in case.Registrations:
    for struct_reg in top_registration.StructureRegistrations:
        if struct_reg.Name.startswith(new_reg_name):
            already_deformed = True
            if not os.path.exists(out_deformation_image):
                struct_reg.ExportDeformedMetaImage(MetaFileName=out_deformation_image)
