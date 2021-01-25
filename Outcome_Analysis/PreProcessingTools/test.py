__author__ = 'Brian M Anderson'
# Created on 1/22/2021
import os
import pandas as pd

excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
             r'\Predicting_Recurrence\RetroAblation.xlsx'
df = pd.read_excel(excel_path, sheet_name='Refined')
for index in df.index:
    MRN = df['MRN'][index]
    df.at[index, 'MRN'] = str(MRN)
df.to_excel(os.path.join('.', 'out.xlsx'), index=0)