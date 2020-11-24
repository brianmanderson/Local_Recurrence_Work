__author__ = 'Brian M Anderson'
# Created on 11/18/2020

import os
import numpy as np
import shutil


def distribute_into_cv(records_path, out_path_base, description, cv_groups=5):
    patient_dictionary = {}
    for file in os.listdir(records_path):
        iteration = file.split('_')[0]
        if iteration not in patient_dictionary:
            patient_dictionary[iteration] = [file]
        else:
            patient_dictionary[iteration] += [file]
    split_key_groups = np.array_split(np.array(list(patient_dictionary.keys())), cv_groups)
    for i in range(cv_groups):
        print('Performing split {} of {}'.format(i + 1, cv_groups))
        out_validation = os.path.join(out_path_base, 'CV_{}'.format(i),
                                      'Validation{}'.format(description))
        out_train = os.path.join(out_path_base, 'CV_{}'.format(i), 'Train{}'.format(description))
        if os.path.exists(out_validation) or os.path.exists(out_train):
            print('Have already split {}. Do not split multiple times!'.format(records_path))
            break
        else:
            os.makedirs(out_validation)
            os.makedirs(out_train)
        for split_index, keys_list in enumerate(split_key_groups):
            out_path = out_train
            if i == split_index:
                out_path = out_validation
            for key in keys_list:
                files = patient_dictionary[key]
                for file in files:
                    shutil.copy(os.path.join(records_path, file), os.path.join(out_path, file))
    return None


if __name__ == '__main__':
    pass
