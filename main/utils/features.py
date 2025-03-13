import numpy as np

def getFeatures(x,y, num_features=10):

    """
    Calculates the features for a given trajectory.

    Args:
        x (numpy array): The x-coordinates of the trajectory.
        y (numpy array): The y-coordinates of the trajectory.
        num_features (int): The number of features to calculate.

    Returns:
        numpy array: The features of the trajectory. Namely:
        1. displacement from origin
        2. z-normalised x
        3. z-normalised y
        4. z-normalised step size
        5. angle
        6. straightness
        7. efficiency
        8. F(z-normalised x)
        9. F(z-normalised y)
        10. F(z-normalised step size)
        where F(k) = log(|Δk|)
    """
        

    size_x = len(x)
    features = np.zeros((size_x, num_features), dtype=np.float32)

    steps = (np.diff(x)**2 + np.diff(y)**2)**0.5 
    disp_from_origin = ((x - x[0])**2 + (y - y[0])**2)**0.5 # this is a feature

    norm_x = (x - np.mean(x))/ np.std(x)# this is a feature
    norm_y = (y - np.mean(y))/ np.std(y)# this is a feature
    norm_stepsize = (steps - np.mean(steps))/ np.std(steps)# this is a feature

    # get angles
    p1 = np.array([x[:-2], y[:-2]])
    p2 = np.array([x[1:-1], y[1:-1]])
    p3 = np.array([x[2:], y[2:]])
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.sum(v1 * v2, axis=0)
    mag_v1 = np.sqrt(np.sum(v1**2, axis=0))
    mag_v2 = np.sqrt(np.sum(v2**2, axis=0))
    product = np.clip(dot_product / (mag_v1 * mag_v2), -1, 1)
    angles = np.arccos(product) # this is a feature
    # angles finished

    norm_displacement_diff = np.diff(norm_x)**2 + np.diff(norm_y)**2
    norm_from_origin_squared = (norm_x - norm_x[0])**2 + (norm_y - norm_y[0])**2
    
    straightness =  norm_from_origin_squared**0.5/ np.insert(np.cumsum(norm_displacement_diff**0.5),0,99,axis=0) # this is a feature
    efficiency = norm_from_origin_squared/ np.insert(np.cumsum(norm_displacement_diff),0,99,axis=0) # this is a feature

    F_x = np.log(np.abs(np.diff(norm_x))) # this is a feature
    F_y = np.log(np.abs(np.diff(norm_y))) # this is a feature
    F_step = np.log(np.abs(np.diff(norm_stepsize))) # this is a feature

    features[:size_x,0] = norm_x
    features[:size_x,1] = norm_y
    features[:size_x,2] = disp_from_origin
    features[:(size_x-1),3] = norm_stepsize
    features[:(size_x-2),4] = angles
    features[:size_x,5] = straightness
    features[:size_x,6] = efficiency
    features[:(size_x-1),7] = F_x
    features[:(size_x-1),8] = F_y
    features[:(size_x-2),9] = F_step

    return features
