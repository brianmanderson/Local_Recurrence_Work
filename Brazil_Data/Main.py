__author__ = 'Brian M Anderson'
# Created on 9/28/2020

'''
First we need to unzip the files and add to directory H:\Brazil_Data
'''


'''
Next we need to add a .dcm tag to each file and updated PatientID
'''

add_tag = False
if add_tag:
    from .Update_Dicom_Patient_ID import update_patient_ID
    path = r'H:\Brazil_Data\Ablation Morfeus Brazilian dataset\Cases'
    update_patient_ID(path=path)
