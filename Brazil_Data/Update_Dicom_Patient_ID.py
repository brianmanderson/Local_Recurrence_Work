__author__ = 'Brian M Anderson'
# Created on 11/3/2020

import os
import pydicom
from threading import Thread
from multiprocessing import cpu_count
from queue import Queue


def change_name_and_pat_id(patient_id, root):
    print(root)
    files = [i for i in os.listdir(root) if i.startswith('IM')]
    for file in files:
        if not file.endswith('.dcm'):
            os.rename(os.path.join(root, file), os.path.join(root, file + '.dcm'))
            file += '.dcm'
        ds = pydicom.read_file(os.path.join(root, file))
        if ds.PatientID == patient_id:  # Already have converted the files in this folder
            break
        ds.PatientID = patient_id
        pydicom.write_file(os.path.join(root, file), ds)
    return None


def worker_def(a):
    q = a[0]
    while True:
        item = q.get()
        if item is None:
            break
        else:
            try:
                change_name_and_pat_id(**item)
            except:
                print('failed?')
            q.task_done()


def update_patient_ID(path=r'H:\Brazil_Data\Ablation Morfeus Brazilian dataset\Cases'):
    patient_dict = {}
    for patient in os.listdir(path):
        patient_id = 'Brazil_dataset_{}'.format(patient.split(' -')[0])
        for root, directory, files in os.walk(os.path.join(path, patient)):
            print(root)
            files = [i for i in files if i.startswith('IM')]
            if files:
                if patient_id not in patient_dict:
                    patient_dict[patient_id] = []
                patient_dict[patient_id].append(root)
    thread_count = int(cpu_count()*.8 - 1)
    q = Queue(maxsize=thread_count)
    a = [q, ]
    threads = []
    for worker in range(thread_count):
        t = Thread(target=worker_def, args=(a,))
        t.start()
        threads.append(t)
    print('Changing everything now')
    for patient_id in patient_dict.keys():
        for root in patient_dict[patient_id]:
            item = {'patient_id': patient_id, 'root': root}
            q.put(item)
    for i in range(thread_count):
        q.put(None)
    for t in threads:
        t.join()


if __name__ == '__main__':
    pass
