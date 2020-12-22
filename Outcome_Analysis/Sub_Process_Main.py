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

args = "python MainDeepLearning.py {} {}"
# args = "python Return_Train_Validation_Generators_TF2.py {}".format(gpu)
model_key = 1
sys.path.append('..')
for _ in range(500):
    print('Running iteration now')
    call(args=args.format(gpu, model_key), shell=True)