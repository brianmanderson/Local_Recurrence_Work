__author__ = 'Brian M Anderson'
# Created on 4/14/2021
"""
Code created to update the exam names found in our Retro_Ablation excel sheet
"""
import pandas as pd
import os

excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
             r'\Predicting_Recurrence\RetroAblation.xlsx'
df = pd.read_excel(excel_path, sheet_name='Refined')
path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
for index in df.index:
    MRN = df['MRN'][index]
    patient_path = os.path.join(path, MRN)
    case = os.listdir(patient_path)[0]
    case_path = os.path.join(patient_path, case)
    for exam_name in ['PreExam', 'Ablation_Exam']:
        exam = df[exam_name][index]
        exam_path = os.path.join(case_path, exam)
        if os.path.exists(exam_path):
            exam_files = os.listdir(exam_path)
            new_exam = [i for i in exam_files if i.startswith('New_Exam')]
            if new_exam:
                new_exam_name = new_exam[0].split('New_Exam_')[-1].split('.txt')[0]
                df.at[index, exam_name] = new_exam_name
                os.rename(exam_path, os.path.join(case_path, new_exam_name))
df.to_excel(os.path.join('.', 'out.xlsx'), index=0)