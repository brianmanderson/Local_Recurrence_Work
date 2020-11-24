__author__ = 'Brian M Anderson'
# Created on 5/7/2020
import sys, os
from subprocess import call

if len(sys.argv) > 1:
    gpu = int(sys.argv[1])
else:
    gpu = 0
print('Running on {}'.format(gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

args = "python MainDeepLearning.py {}".format(gpu)
# args = "python Return_Train_Validation_Generators_TF2.py {}".format(gpu)
num_models = 18
iterations = 3
for _ in range(100):
    print('Running iteration now')
    call(args=args, shell=True)