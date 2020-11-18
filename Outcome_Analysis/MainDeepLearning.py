__author__ = 'Brian M Anderson'
# Created on 11/18/2020
'''
All parts here are in the DeepLearningTools folder
'''
import sys, os

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

workondeeplearning = True
if workondeeplearning:
    from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators
    xxx = return_generators()