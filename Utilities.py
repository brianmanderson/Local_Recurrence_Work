__author__ = 'Brian M Anderson'
# Created on 12/30/2019
from Dicom_RT_and_Images_to_Mask.Image_Array_And_Mask_From_Dicom_RT import Dicom_to_Imagestack, plot_scroll_Image, plt, np


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
    polar_points[:,2] = np.arctan2(zxy[:,2], zxy[:,1])
    # polar_points[:, 2][polar_points[:, 2] < 0] += 2 * np.pi  # let them range +0 to +pi and -0 to -pi
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
    return polar_coordinates


def define_cone(polar_cords_base, centroid_of_ablation_recurrence,liver_recurrence, spacing, margin=100, min_max=False,
                margin_rad=np.deg2rad(5)):
    '''
    :param polar_cords_base: polar coordinates from ablation_recurrence centroid to recurrence, come in [phi, theta]
    where theta ranges from 0 to pi and -0 to -pi
    :param centroid_of_ablation_recurrence: centroid of ablation recurrence
    :param liver_recurrence: shape used to make output
    :param margin: how far would you like to look, in mm
    :param margin_rad: degrees of wiggle allowed, recommend 5 degrees (in radians)
    :return:
    '''
    polar_cords_base = polar_cords_base.astype('float16')
    if polar_cords_base.shape[1] == 3:
        polar_cords_base = polar_cords_base[:,1:]
    cone_cords_base = create_distance_field(np.ones(liver_recurrence.shape),origin=centroid_of_ablation_recurrence,spacing=spacing)
    cone_cords_base = np.round(cone_cords_base,3).astype('float16')
    output = np.zeros(cone_cords_base.shape[0])
    positive_polar_indexes = polar_cords_base[:,1] >= 0
    negative_polar_indexes = polar_cords_base[:,1] <= 0
    positive_cord_indexes = cone_cords_base[:,2] >= 0
    negative_cord_indexes = cone_cords_base[:,2] <= 0
    for polar_indxes, cord_indexes in zip([positive_polar_indexes, negative_polar_indexes],
                                          [positive_cord_indexes,negative_cord_indexes]):
        polar_cords = polar_cords_base[polar_indxes]
        cone_cords = cone_cords_base[cord_indexes]
        if not np.any(polar_cords) or not np.any(cone_cords):
            continue
        mask = np.zeros(output[cord_indexes].shape)
        min_phi, max_phi, min_theta, max_theta = min(polar_cords[..., 0]), max(polar_cords[..., 0]), min(
            polar_cords[..., 1]), max(polar_cords[..., 1])
        min_phi, max_phi, min_theta, max_theta = min_phi - margin_rad, max_phi + margin_rad, min_theta - margin_rad, \
                                                 max_theta + margin_rad
        if min_max:
            vals = np.where(
                (cone_cords[:, 1] >= min_phi) & (cone_cords[:, 1] <= max_phi) & (cone_cords[:, 2] >= min_theta)
                & (cone_cords[:, 2] <= max_theta))
            mask[vals[0]] = 1
            output[cord_indexes] = mask
        else:
            vals = np.where((cone_cords[:, 1] >= min_phi) & (cone_cords[:, 1] <= max_phi) & (cone_cords[:,0] < margin) &
                            (cone_cords[:, 2] >= min_theta) & (cone_cords[:, 2] <= max_theta))
            cone_cords_reduced = cone_cords[vals[0]][:,1:]
            del cone_cords
            difference = cone_cords_reduced[:,None] - polar_cords
            del polar_cords
            min_dif = np.min(difference**2,axis=1) # find the minimum difference for each point against the ablation region
            del difference
            total_dif = np.sqrt(np.sum(min_dif,axis=1))
            del min_dif
            dif_vals = np.where(total_dif<margin_rad) # Allow wiggle
            mask = np.zeros(output[cord_indexes].shape)
            mask[vals[0][dif_vals[0]]] = 1
            output[cord_indexes] = mask
            del dif_vals
        del vals
    output = np.reshape(output,liver_recurrence.shape) # This is now a cone including the recurrence site
    return output


def main():
    pass


if __name__ == '__main__':
    main()