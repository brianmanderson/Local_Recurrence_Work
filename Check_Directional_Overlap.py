__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
import SimpleITK as sitk
import time


def cartesian_to_polar(zxy):
    '''
    :param xyz: array of zxy cooridnates
    :return: polar coordinates in the form of: radius, rotation away from the z axis (phi), and rotation from the
    negative y axis (theta)
    '''
    # xyz = np.stack([x, y, z], axis=-1)
    input_shape = zxy.shape
    reshape = False
    if len(input_shape) > 2:
        reshape = True
        zxy = np.reshape(zxy,[np.prod(zxy.shape[:-1]),3])
    polar_points = np.empty(zxy.shape)
    # ptsnew = np.hstack((xyz, np.empty(xyz.shape)))
    xy = zxy[:,2]**2 + zxy[:,1]**2
    polar_points[:,0] = np.sqrt(xy + zxy[:,0]**2)
    polar_points[:,1] = np.arctan2(np.sqrt(xy), zxy[:,0])  # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    polar_points[:,2] = np.arctan2(zxy[:,1], zxy[:,2])
    polar_points[:, 2][polar_points[:, 2] < 0] += 2 * np.pi  # Make them all 0 to 2 pi for theta
    if reshape:
        polar_points = np.reshape(polar_points,input_shape)
    return polar_points


def polar_to_cartesian(polar):
    '''
    :param polar: in the form of radius, angle away from z axis, and angle from the negative y axis
    :return: z, x, y intensities as differences from the origin
    '''
    cartesian_points = np.empty(polar.shape)
    from_x = polar[:,2]
    xy_plane = np.sin(polar[:,1])*polar[:,0]
    cartesian_points[:,0] = np.cos(polar[:,1])*polar[:,0]
    cartesian_points[:,1] = np.cos(from_x)*xy_plane
    cartesian_points[:,2] = np.sin(from_x)*xy_plane
    return cartesian_points


def create_ray_spectrum(image,origin, radius=50,spacing=(0.975,0.975,5),theta_range=range(0,360,1),
                        phi_range=range(0,180,1)):
    '''
    :param image:
    :param origin:
    :param radius: in mm
    :param spacing: spacing in [x, y, z] coorindates
    :param theta_range: array of values in theta range
    :param phi_range: array of values in the phi range
    :return:
    '''
    temp_image = np.zeros(image.shape)
    for degree_theta in theta_range:
        print(degree_theta)
        theta_rad = np.deg2rad(degree_theta)
        for degree_phi in phi_range:
            phi_rad = np.deg2rad(degree_phi)
            keep = False
            for rad in range(1, radius):
                new_x = int(origin[1] + rad*np.cos(theta_rad)*np.sin(phi_rad))
                new_y = int(origin[2] + rad * np.sin(theta_rad) * np.sin(phi_rad))
                new_z = int(origin[0] + rad * np.cos(phi_rad))
                rad_made = np.sqrt(((origin[0] - new_z) * spacing[2]) ** 2 + ((origin[1] - new_x) * spacing[1]) ** 2 +
                                   ((origin[2] - new_y) * spacing[2]) ** 2)
                if rad_made > radius:
                    break
                try:
                    temp_image[new_z, new_x, new_y] = 2
                    if image[new_z, new_x, new_y] == 1:
                        keep = True
                except:
                    break
            if keep:
                temp_image[temp_image == 2] = 1
            else:
                temp_image[temp_image == 2] = 3
    np.save(os.path.join('..','saved2.npy'),temp_image)
    return temp_image


def create_distance_field(image,origin, spacing=(0.975,0.975,5.0)):
    '''
    :param image:
    :param origin:
    :param spacing:
    :return: polar coordinates, in: [radius, rotation away from the z axis (phi), and rotation from the y axis (theta)]
    '''
    array_of_points = np.transpose(np.asarray(np.where(image==1)),axes=(1,0))
    spacing_aranged = np.asarray([spacing[2],spacing[0],spacing[1]])
    differences = (array_of_points - origin)*spacing_aranged
    polar_coordinates = cartesian_to_polar(differences)
    image_2 = np.zeros(image.shape)
    k = polar_to_cartesian(polar_coordinates)
    k += origin
    k = round(k)
    image_2[k] = 1
    return polar_coordinates


'''
This should have two parts... first, check the recurrence image for what direction
the recurrence occurred
Then, look at the post-treatment image and see if there was 5 mm margin existing in that direction
'''
new_images = np.load(os.path.join('..', 'saved2.npy'))
images_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
excel_file = os.path.join('..','Data','Post_treatment_and_Recurrence_info.xlsx')
data = pd.read_excel(excel_file)
MRNs = data['MRN']
for index in range(len(MRNs)):
    index = 1
    MRN = str(data['MRN'][index])
    Secondary = data['Secondary'][index]
    Recurrence = data['Recurrence'][index]
    recurrence_path = os.path.join(images_path,MRN,Recurrence)
    recurrence_reader = Dicom_to_Imagestack(arg_max=False,Contour_Names=['Liver','recurrence','Ablation_Recurrence',
                                                                         'Ablation','GTV_Exp_5mm_outside_Ablation'])
    recurrence_reader.Make_Contour_From_directory(recurrence_path)

    mask = recurrence_reader.mask
    liver = mask[...,1]
    # centroid_of_recurrence = center_of_mass(mask[...,2])
    centroid_of_ablation = np.asarray(center_of_mass(mask[...,3]))
    recurrence = recurrence_reader.mask[...,2]
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    recurrence = np.zeros([10,200,200])
    recurrence[5,:50,:50] = 1
    centroid_of_ablation = (5,100,100)
    liver = np.zeros(recurrence.shape)
    spacing = (1,1,1)
    polar_cords = create_distance_field(recurrence,origin=centroid_of_ablation, spacing=spacing)
    min_phi, max_phi, min_theta, max_theta = min(polar_cords[...,1]), max(polar_cords[...,1]), min(polar_cords[...,2]),\
                                             max(polar_cords[...,2])
    cone_cords = create_distance_field(np.ones(liver.shape),origin=centroid_of_ablation,spacing=spacing)
    output = np.zeros(cone_cords.shape[0])
    vals = np.where((cone_cords[:,1]>min_phi)&(cone_cords[:,1]<max_phi)&(cone_cords[:,2]>min_theta)&(cone_cords[:,2]<max_theta))
    output[vals[0]] = 1
    output = np.reshape(output,liver.shape)
    max_radius = np.max(polar_cords[:,0])
    start = time.time()
    zz, xx, yy = np.mgrid[0:recurrence.shape[0],0:recurrence.shape[1],0:recurrence.shape[2]]
    circle = np.sqrt(((zz - centroid_of_ablation[0])*spacing[-1]) ** 2 + ((xx - centroid_of_ablation[1]) * spacing[0]) ** 2 + \
             ((yy - centroid_of_ablation[2])*spacing[2]) ** 2)
    out = np.zeros(recurrence.shape)
    out[circle<max_radius] = 1
    print(time.time()-start)
    break
xxx = 1
