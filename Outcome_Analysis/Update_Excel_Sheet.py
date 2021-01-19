__author__ = 'Brian M Anderson'
# Created on 1/19/2021
import os
import pandas as pd

path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence'
file_path = os.path.join(path, 'RetroAblation - Copy.xlsx')
current_df = pd.read_excel(file_path)
status_df = pd.read_excel(os.path.join(path, 'Copy of Retro_ablation_2019_212 cases_FU.xlsx'))

lacking_status = current_df.loc[pd.isnull(current_df['Status'])]
indexes = lacking_status.index.values
for index in indexes:
    try:
        MRN = int(current_df['MRN'][index])
    except:
        continue
    pre_exam = current_df['PreExam'][index].replace(' ', '')
    post_exam = current_df['Ablation_Exam'][index].replace(' ', '')
    segment = current_df['Segment of lesion'][index]
    lesion_number = current_df['Lesion number'][index]
    status = status_df.loc[(status_df['MRN'] == MRN) & (status_df['Pre-ablation Exam'] == pre_exam) &
                           (status_df['Post-ablation Exam Date'] == post_exam) &
                           (status_df['Segment of lesion'] == segment)]
    if status.shape[0] > 1:
        status = status.loc[status['Lesion number'] == lesion_number]
    if status.shape[0] == 1:
        lesion_status = status['Tumor status after ablation (1:Residual, 2:Local progression, 3:none)'].values[0]
        lesion_number = status['Lesion number'].values[0]
        current_df.at[index, 'Status'] = int(lesion_status)
        current_df.at[index, 'Lesion number'] = lesion_number
    elif status.shape[0] == 0:
        print('None')
    else:
        print('two values..{}'.format(MRN))
current_df.to_excel(file_path, index=0)
