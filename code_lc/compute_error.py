import numpy as np

def compute_error_single(test_corrd,ground_truth_frame,threshold,test_point):
    num_of_points=len(ground_truth_frame)
    if num_of_points == 68:
        interocular_distance = np.linalg.norm(ground_truth_frame[36,:]-ground_truth_frame[45,:])
    elif num_of_points == 51:
        interocular_distance = np.linalg.norm(ground_truth_frame[19,:]-ground_truth_frame[28,:])
    #print(interocular_distance)
    error=np.linalg.norm(test_corrd-ground_truth_frame[test_point])/interocular_distance
    return (0 if error<threshold else 1, True if error<threshold else False,interocular_distance, error)

def compute_error_all_point(test_corrd,ground_truth_frame,threshold,interocular_distance=None):
    num_of_points=len(ground_truth_frame)
    if type(interocular_distance)==type(None):
        if num_of_points==106:
            interocular_distance = np.linalg.norm(ground_truth_frame[66,:]-ground_truth_frame[79,:])
        elif num_of_points == 68:
            interocular_distance = np.linalg.norm(ground_truth_frame[36,:]-ground_truth_frame[45,:])
        elif num_of_points == 7:
            interocular_distance = np.linalg.norm(ground_truth_frame[0,:]-ground_truth_frame[3,:])
    #print(interocular_distance)
    error=np.mean(np.linalg.norm(test_corrd-ground_truth_frame,axis=-1))/interocular_distance
    #print(np.linalg.norm(test_corrd-ground_truth_frame,axis=-1).shape)
    error_array=np.linalg.norm(test_corrd-ground_truth_frame,axis=-1)/interocular_distance
    return (0 if error<threshold else 1, True if error<threshold else False,interocular_distance, error,error_array)


def compute_error(ground_truth_all, detected_points_all,test_point=None,interocular_distance=None):
    '''
    compute_error
    compute the average point-to-point Euclidean error normalized by the
    inter-ocular distance (measured as the Euclidean distance between the
        outer corners of the eyes)

   Inputs:
          grounth_truth_all, size: num_of_images x num_of_points x 2
          detected_points_all, size: num_of_images x num_of_points x 2
   Output:
          error_per_image, size: num_of_images
    '''


    num_of_images = ground_truth_all.shape[0]
    num_of_points = ground_truth_all.shape[1]

    error_per_image = []

    for i in(range(num_of_images)):
        detected_points=detected_points_all[i,:,:]
        ground_truth_points=ground_truth_all[i,:,:]
        if type(interocular_distance)==type(None):
            if num_of_points == 68:
                interocular_distance = np.linalg.norm(ground_truth_points[36,:]-ground_truth_points[45,:])
            elif num_of_points == 51:
                interocular_distance = np.linalg.norm(ground_truth_points[19,:]-ground_truth_points[28,:])
        norm_sum=0.0
        if type(test_point)!=type(None):
            norm_sum = np.linalg.norm(detected_points[test_point,:]-ground_truth_points[test_point,:])
            error_per_image.append(norm_sum/(interocular_distance))
        else:
            for j in range(num_of_points):
                norm_sum = norm_sum+np.linalg.norm(detected_points[j,:]-ground_truth_points[j,:])
            error_per_image.append(norm_sum/(num_of_points*interocular_distance))
    return error_per_image
