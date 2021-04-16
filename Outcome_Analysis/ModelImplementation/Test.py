__author__ = 'Brian M Anderson'
# Created on 4/12/2021
import SimpleITK as sitk
import os
import numpy as np


path = r'H:\CreatingQualitativeMaps\Pat'
exam = sitk.ReadImage(os.path.join(path, 'Exam.mhd'))
exam_size = exam.GetSize()
visual = np.random.rand(exam_size[2], exam_size[1], exam_size[0], 1)
visual *= 255
visual = visual.astype('int16')
visual_handle = sitk.GetImageFromArray(visual)
visual_handle.SetSpacing(exam.GetSpacing())
visual_handle.SetDirection(exam.GetDirection())
visual_handle.SetOrigin(exam.GetOrigin())
visual_handle.SetMetaData("ElementNumberOfChannels", '1')
sitk.WriteImage(visual_handle, os.path.join(path, 'Test.mhd'))


colorTable = {0: System.Drawing.Color.FromArgb(255, 0, 0, 0),
14: System.Drawing.Color.FromArgb(255, 0, 0, 255),
20: System.Drawing.Color.FromArgb(255, 127, 255, 0),
32: System.Drawing.Color.FromArgb(255, 255, 255, 0),
36: System.Drawing.Color.FromArgb(255, 255, 127, 0),
100: System.Drawing.Color.FromArgb(255, 255, 0, 0)}
