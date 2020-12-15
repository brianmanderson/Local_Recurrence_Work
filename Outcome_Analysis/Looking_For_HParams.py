__author__ = 'Brian M Anderson'
# Created on 12/14/2020
import os
from matplotlib import pyplot as plt
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import plugin_event_accumulator
import tensorflow as tf


path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Test_Index\Model_Index_252'
for root, folders, files in os.walk(path):
    event_files = [i for i in files if i.find('event') == 0]
    if event_files:
        for file in event_files:
            k2 = event_accumulator.EventAccumulator(path=root, size_guidance=0)
            k2.Reload()
            k2.SummaryMetadata('_hparams_/session_start_info')
            k = summary_iterator(os.path.join(root, file))
            temp_dictionary = {}
            for event in k:
                for value in event.summary.value:
                    if value.tag not in temp_dictionary:
                        temp_dictionary[value.tag] = []
                    temp_dictionary[value.tag].append(value.simple_value)
            xxx = 1