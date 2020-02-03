__author__ = 'Brian M Anderson'
# Created on 12/30/2019
from scipy.ndimage.measurements import center_of_mass
from Ray_Tracing.Utilities import *


def main():
    recurrence = np.zeros([10,200,200])
    recurrence[5,20:60,20:60] = 1
    ablation = np.zeros([10,200,200])
    ablation[5,125:180,100:150] = 1
    centroid_of_ablation = np.asarray(center_of_mass(ablation))
    liver = np.zeros(recurrence.shape)
    spacing = (5,.97,.97)
    polar_cords = create_distance_field(recurrence,origin=centroid_of_ablation, spacing=spacing)
    polar_cords = np.round(polar_cords,6)
    output = define_cone(polar_cords, centroid_of_ablation, liver, spacing, margin=99999)
    output[output==1] = 3
    output[recurrence==1] = 1
    output[np.where((ablation==1)&(output!=3))] = 2
    xxx = 1


if __name__ == '__main__':
    main()
