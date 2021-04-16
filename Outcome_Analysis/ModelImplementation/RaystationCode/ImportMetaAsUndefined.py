__author__ = 'Brian M Anderson'
# Created on 4/13/2021
from connect import *
import os
import clr
import sys

from connect import *

import platform
if platform.python_implementation() != "CPython":
  print("Python interpreter should be CPython, but is currently %s" % (platform.python_implementation()))
  sys.exit()

clr.AddReference("PresentationFramework")
clr.AddReference("PresentationCore")
import System
from System import Windows
from System.Windows import MessageBox, Visibility, Window, Application


from System.Windows.Markup import XamlReader
from System.IO import StringReader
from System.Xml import XmlReader
from System.Threading import Thread, ThreadStart, ApartmentState

from System.Windows import LogicalTreeHelper as lth


def import_meta():
    path = r'H:\CreatingQualitativeMaps\Pat'
    case = get_current("Case")
    case.ImportMetaImageToCurrentPatientAsUndefined(ExaminationName='Qualitative',
                                                    MetaFileName=os.path.join(path, 'Out_Gradient.mhd'),
                                                    Rescale=None, FrameOfReference='')


def add_colormap():
    case = get_current("Case")
    colorTable = {0: System.Drawing.Color.FromArgb(255, 0, 0, 0),
                  10: System.Drawing.Color.FromArgb(255, 0, 0, 0),
                  14: System.Drawing.Color.FromArgb(255, 0, 0, 255),
                  20: System.Drawing.Color.FromArgb(255, 127, 255, 0),
                  32: System.Drawing.Color.FromArgb(255, 255, 255, 0),
                  36: System.Drawing.Color.FromArgb(255, 255, 127, 0),
                  100: System.Drawing.Color.FromArgb(255, 255, 0, 0)}
    isDiscrete = False
    colorMapReferenceType = "MaxValue"
    isLowValuesClipped = False
    isHighValuesClipped = False
    presentationType = "Absolute"
    auxiliaryUnit = "_"

    exam_name = 'Qualitative'
    case.Examinations[exam_name].SetPrimary()

    case.Examinations[exam_name].SetImageSpecificColorMap(ColorTable=colorTable, IsDiscrete=isDiscrete,
                                                          IsLowValuesClipped=isLowValuesClipped,
                                                          IsHighValuesClipped=isHighValuesClipped,
                                                          ReferenceValue=1.0,
                                                          ColorMapReferenceType=colorMapReferenceType,
                                                          PresentationType=presentationType,
                                                          AuxiliaryUnit=auxiliaryUnit)


def main():
    # import_meta()
    add_colormap()


if __name__ == '__main__':
    main()
