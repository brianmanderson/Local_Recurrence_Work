"# Local_Recurrence_Work" 

This is work done to create a pseudo-raytrace to identify if the 5mm minimum ablation margin is occuring in the same region as the local recurrence.

The 'Proof_Of_Concept_Cross' takes two rois, defined as 'Test_Ablation' and 'Test_Cross', shown in the 'Picture_With_Cross_Explanation' as the yellow cylinder and blue cross.
The min_max picture shows a very fast approach that only looks at the minimum/maximum phi/theta in spherical coorindates to draw a cone of influence.
The not min_max picture show a slightly slower approach to identify fine edges.

It is recommended not to use a margin greater than 100, as the process becomes increasingly slow with more points.
It is recommended to use a margin_rad of at least 5 degrees.


    import numpy as np
    recurrence_reader = Dicom_to_Imagestack(arg_max=False, Contour_Names=['Test_Ablation','Test_Cross'])
    recurrence_reader.Make_Contour_From_directory(recurrence_path)

    mask = recurrence_reader.mask
    ablation_base = mask[...,1]
    cross_base = mask[...,2]
    centroid_of_ablation_recurrence = np.asarray(center_of_mass(ablation_base)) # Get the centroid of the ablation margin
    spacing = recurrence_reader.annotation_handle.GetSpacing()
    labels = morphology.label(cross_base, neighbors=4)  # Could have multiple recurrence sites
    output = np.zeros(cross_base.shape)
    output_recurrence = np.expand_dims(output, axis=-1)
    output_recurrence = np.repeat(output_recurrence, repeats=2, axis=-1)
    for label_value in range(1, np.max(labels) + 1): # For each potential recurrence site, draw a cone
        recurrence = np.zeros(cross_base.shape)
        recurrence[labels == label_value] = 1
        polar_cords = create_distance_field(recurrence, origin=centroid_of_ablation_recurrence, spacing=spacing)
        polar_cords = np.round(polar_cords, 3).astype('float16')

        polar_cords = polar_cords[:, 1:]
        '''
        We now have the min/max phi/theta for pointing the recurrence_ablation site to the recurrence

        Now, we take those coordinates and see if, with the ablation to minimum ablation site overlap

        Note: This will turn a star shape into a square which encompasses the star!
        '''
        output_recurrence[..., 1] += define_cone(polar_cords, centroid_of_ablation_recurrence, cross_base,
                                                 spacing, margin=75, min_max=False, margin_rad=np.deg2rad(5))
    recurrence_reader.with_annotations(output_recurrence, output_dir=os.path.join(recurrence_path, 'new_RT'),
                                       ROI_Names=['cone_cross_fixed'])
 ![alt test](Picture_With_Cross_Explanation.jpg)
