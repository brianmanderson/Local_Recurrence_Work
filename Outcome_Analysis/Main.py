__author__ = 'Brian M Anderson'
# Created on 9/28/2020

'''
First thing we need to do is create segmentations of liver, GTV, and ablation on our exams
'''

'''
Run Create_Liver_Contours
Create_Liver_Contours.py
'''

'''
Run Create_Disease_Ablation_Contours
Create_Disease_Ablation_Contours.py
'''

'''
Run Create_ROI_Names
This is where we create the contour names 'Retro_GTV', 'Retro_Ablation', and 'Retro_GTV_Recurred'
The ROI will help train the model for outcomes
'''

'''
Export the DICOM and RT
Export_Patients.py
'''

'''
Register the images about the liver and export
'''
