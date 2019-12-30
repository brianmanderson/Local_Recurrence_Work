__author__ = 'Brian M Anderson'
# Created on 12/30/2019
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
from Utilities import *


def main():
    recurrence = np.zeros([10,200,200])
    recurrence[1,20:60,20:60] = 1
    ablation = np.zeros([10,200,200])
    ablation[5,125:180,100:150] = 1
    centroid_of_ablation = np.asarray(center_of_mass(ablation))
    liver = np.zeros(recurrence.shape)
    spacing = (5,.97,.97)
    polar_cords = create_distance_field(recurrence,origin=centroid_of_ablation, spacing=spacing)
    polar_cords = np.round(polar_cords,4)
    min_phi, max_phi, min_theta, max_theta = min(polar_cords[...,1]), max(polar_cords[...,1]), min(polar_cords[...,2]),\
                                             max(polar_cords[...,2])
    cone_cords = create_distance_field(np.ones(liver.shape),origin=centroid_of_ablation,spacing=spacing)
    cone_cords = np.round(cone_cords,4)
    output = np.zeros(cone_cords.shape[0])
    vals = np.where((cone_cords[:,1]>=min_phi)&(cone_cords[:,1]<=max_phi)&(cone_cords[:,2]>=min_theta)&(cone_cords[:,2]<=max_theta))
    output[vals[0]] = 3
    output = np.reshape(output,liver.shape)
    output[recurrence==1] = 1
    output[np.where((ablation==1)&(output!=3))] = 2
    xxx = 1


if __name__ == '__main__':
    main()
