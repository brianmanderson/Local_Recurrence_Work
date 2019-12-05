__author__ = 'Brian M Anderson'
# Created on 12/5/2019
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np
from scipy.ndimage.measurements import center_of_mass
import SimpleITK as sitk
import time


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
    np.save(os.path.join('.','saved.npy'),temp_image)
    return temp_image


'''
This should have two parts... first, check the recurrence image for what direction
the recurrence occurred
Then, look at the post-treatment image and see if there was 5 mm margin existing in that direction
'''
new_images = np.load(os.path.join('.', 'saved.npy'))
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
    centroid_of_ablation = center_of_mass(mask[...,3])
    recurrence = recurrence_reader.mask[...,2]
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    start = time.time()
    new_mask = create_ray_spectrum(recurrence,centroid_of_ablation,100, spacing=spacing)
    print(time.time()-start)
    break

    xxx = 1
xxx = 1
