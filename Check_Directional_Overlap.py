__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
import SimpleITK as sitk
import time


def cartesian_to_polar(xyz):
    '''
    :param x: x_values in single array
    :param y: y_values in a single array
    :param z: z_values in a single array
    :return: polar coordinates in the form of: radius, rotation away from the z axis, and rotation from the y axis
    '''
    # xyz = np.stack([x, y, z], axis=-1)
    input_shape = xyz.shape
    reshape = False
    if len(input_shape) > 2:
        reshape = True
        xyz = np.reshape(xyz,[np.prod(xyz.shape[:-1]),3])
    polar_points = np.empty(xyz.shape)
    # ptsnew = np.hstack((xyz, np.empty(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    polar_points[:,0] = np.sqrt(xy + xyz[:,2]**2)
    polar_points[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    polar_points[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    if reshape:
        polar_points = np.reshape(polar_points,input_shape)
    return polar_points


def polar_to_cartesian(polar_xyz):
    '''
    :param polar_xyz: in the form of radius, elevation away from z axis, and elevation from y axis
    :return: x, y, and z intensities
    '''
    cartesian_points = np.empty(polar_xyz.shape)
    from_y = polar_xyz[:,2]
    xy_plane = np.sin(polar_xyz[:,1])*polar_xyz[:,0]
    cartesian_points[:,2] = np.cos(polar_xyz[:,1])*polar_xyz[:,0]
    cartesian_points[:,0] = np.sin(from_y)*xy_plane
    cartesian_points[:,1] = np.cos(from_y)*xy_plane
    return cartesian_points


def create_ray_spectrum(image,origin, radius=50,spacing=(0.975,0.975,5)):
    '''
    :param image:
    :param origin:
    :param radius: in mm
    :param spacing: spacing in [x, y, z] coorindates
    :return:
    '''
    temp_image = np.zeros(image.shape)
    for degree_theta in range(0, 360, 1):
        print(degree_theta)
        theta_rad = np.deg2rad(degree_theta)
        for degree_phi in range(0,180,1):
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
    np.save(os.path.join('..','saved.npy'),temp_image)
    return temp_image


def create_distance_field(image,origin, spacing=(0.975,0.975,5.0)):
    array_of_points = np.transpose(np.asarray(np.where(image==1)),axes=(1,0))
    differences = array_of_points - origin
    xxx = 1
    return None

'''
This should have two parts... first, check the recurrence image for what direction
the recurrence occurred
Then, look at the post-treatment image and see if there was 5 mm margin existing in that direction
'''
new_images = np.load(os.path.join('..', 'saved.npy'))
images_path = r'K:\Morfeus\BMAnderson\CNN\Data\Data_Liver\Recurrence_Data\Images'
excel_file = os.path.join('..','Data','Post_treatment_and_Recurrence_info.xlsx')
data = pd.read_excel(excel_file)
MRNs = data['MRN']
for index in range(len(MRNs)):
    MRN = str(data['MRN'][index])
    Secondary = data['Secondary'][index]
    Recurrence = data['Recurrence'][index]
    recurrence_path = os.path.join(images_path,MRN,Recurrence)
    recurrence_reader = Dicom_to_Imagestack(arg_max=False,Contour_Names=['Liver','recurrence','Ablation'])
    recurrence_reader.Make_Contour_From_directory(recurrence_path)

    mask = recurrence_reader.mask
    liver = mask[...,1]
    mask[liver == 0] = 0  # Bring everything to be within the liver
    # centroid_of_recurrence = center_of_mass(mask[...,2])
    centroid_of_ablation = np.asarray(center_of_mass(mask[...,3]))
    recurrence = recurrence_reader.mask[...,2]
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    create_distance_field(recurrence,origin=centroid_of_ablation, spacing=spacing)
    start = time.time()
    new_mask = create_ray_spectrum(recurrence,centroid_of_ablation,100, spacing=spacing)
    print(time.time()-start)
    break
xxx = 1
