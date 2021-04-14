__author__ = 'Brian M Anderson'
# Created on 4/14/2021
"""
This is code created to update the exam names in our excel sheets and folder paths
Exam names changed after the migration from General database to Brocklab database

This script will create a text-file in each folder with the SeriesInstanceUID as well as what exam it used to be called
"""
import os
import pydicom


def main():
    path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    for root, folders, files in os.walk(path):
        dicom_files = [i for i in files if i.endswith('.dcm')]
        if len(dicom_files) > 20:
            print(root)
            previous_exam_name = os.path.split(root)[-1]
            fid = open(os.path.join(root, 'Old_Exam_{}.txt'.format(previous_exam_name)), 'w+')
            fid.close()
            for file in dicom_files:
                ds = pydicom.read_file(os.path.join(root, file))
                series_instance_uid = ds.SeriesInstanceUID
                fid = open(os.path.join(root, 'SeriesInstanceUID.txt'), 'w+')
                fid.write(series_instance_uid)
                fid.close()
                break
    return None


if __name__ == '__main__':
    main()
