
import numpy as np
import ruptures as rpt
from scipy.ndimage import median_filter

"""
Postprocessing functions.
"""

def median_filter_1d(arr, window=3):
    """
    Applies a median filter to a 1D array.

    Args:
        arr (numpy array): The input array.

    Returns:
        numpy array: The filtered array.
    """

    filtered_arr = median_filter(arr, size=window, mode='nearest')
    
    # challenge constraints - changepoints can only occur after 3 consecutive values, so we know
    # the first and last values of the filtered array will be the same as the 3rd and 2nd values
    # comment this if not needed
    filtered_arr[0], filtered_arr[1] = filtered_arr[2], filtered_arr[2]
    filtered_arr[-1], filtered_arr[-2] = filtered_arr[-3], filtered_arr[-3]

    return filtered_arr


def smooth_series(series, lower_limit=None, upper_limit=None, threshold=0.01):
    """
    Smooths a series by replacing consecutive similar values with their average.

    Args:
        series (numpy array): The input series.
        lower_limit (float, optional): Lower limit for the series values. Values below this limit will be replaced.
        upper_limit (float, optional): Upper limit for the series values. Values above this limit will be replaced.

    Returns:
        numpy array: The smoothed series.
    """

    if lower_limit is not None:
        series[series <= lower_limit] = lower_limit
    
    if upper_limit is not None:
        series[series >= upper_limit] = upper_limit

    differences = np.abs(np.diff(series))
    change_indices = np.where(differences > threshold)[0]
    smoothed_series = np.zeros(len(series))

    if len(change_indices) == 0:
        # there are no changes in the series, replace all values with the mean
        smoothed_series[:] = np.mean(series)
    else:
        # there are changes in the series, replace the values between the changes with the mean
        change_indices = change_indices + 1
        change_indices = np.concatenate(([0], change_indices, [len(series)]))

        for i in range(len(change_indices) - 1):
            smoothed_series[change_indices[i]:change_indices[i+1]] = np.mean(series[change_indices[i]:change_indices[i+1]])

    return smoothed_series


def alpha_cps_function(pred_series, penalty, threshold):
    
    """
    Finds changepoints in the alpha series using the Pelt algorithm with an L2 cost function.

    Args:
        pred_series (numpy array): The input series.
        penalty (float): The penalty parameter for the Pelt algorithm.

    Returns:
        list: The list of changepoints.
    """
        
    if np.max(pred_series) != np.min(pred_series):
        pred_series_scaled = (pred_series - np.min(pred_series)) / (np.max(pred_series) - np.min(pred_series))
    else:
        pred_series_scaled = np.ones(len(pred_series)) * 0.5 #scale them to default value of 0.5

    algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(pred_series_scaled)
    cps = algo.predict(pen=penalty)
    
    # cps = [cp for cp in cps if cp > 2 and cp < len(pred_series)-2] #challenge constraint - changepoints can only occur after min size of 3. This should be ok with min_size=3 in the Pelt algo but just to be sure
    # cps.append(len(pred_series))

    cps = [0] + cps # ruptures returns a list

    # remove changepoints that are too close to each other in mean value
    remove = []
    for i in range(1, len(cps) - 1):
        left = pred_series[cps[i - 1]:cps[i]]
        right = pred_series[cps[i]:cps[i + 1]]
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        
        if abs(left_mean - right_mean) <= threshold:
            remove.append(cps[i])
    
    cps = [cp for cp in cps if cp not in remove]
    cps.pop(0)
    return cps

def k_cps_function(pred_series, penalty, threshold):
    
    """
    Finds changepoints in a series using the Pelt algorithm with an L2 cost function.

    Args:
        pred_series (numpy array): The input series.
        penalty (float): The penalty parameter for the Pelt algorithm.

    Returns:
        list: The list of changepoints.
    """
        
    if np.max(pred_series) != np.min(pred_series):
        pred_series_scaled = (pred_series - np.min(pred_series)) / (np.max(pred_series) - np.min(pred_series))
    else:
        pred_series_scaled = np.ones(len(pred_series)) * 0.5 #scale them to default value of 0.5

    algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(pred_series_scaled)
    cps = algo.predict(pen=penalty)
    
    # cps = [cp for cp in cps if cp > 2 and cp < len(pred_series)-2]
    # cps.append(len(pred_series))

    cps = [0] + cps

    # remove changepoints that are too close to each other in mean value
    remove = []

    for i in range(1, len(cps) - 1):

        left = pred_series[cps[i - 1]:cps[i]]
        right = pred_series[cps[i]:cps[i + 1]]
        left_mean = np.mean(left)
        right_mean = np.mean(right)

        if abs(left_mean - right_mean) < threshold:
            remove.append(cps[i])
    
    cps = [cp for cp in cps if cp not in remove]
    cps.pop(0)

    return cps


def replace_short_sequences(arr, min_length=3):

    """
    Replaces short sequences of repeated values in an array with surrounding values.

    Args:
        arr (list or numpy array): The input array.
        min_length (int, optional): The minimum length of a sequence to be replaced.

    Returns:
        list or numpy array: The modified array.
    """

    n = len(arr)
    i = 0
    
    while i < n:
        start = i
        while i < n - 1 and arr[i] == arr[i + 1]:
            i += 1
        length = i - start + 1
        
        # If the sequence is shorter than min_length, replace with surrounding values
        if length < min_length:
            if start == 0:
                replacement_value = arr[i + 1] if i + 1 < n else arr[start]
            elif i == n - 1:
                replacement_value = arr[start - 1]
            else:
                replacement_value = arr[start - 1] if arr[start - 1] == arr[i + 1] else arr[start - 1]
                
            arr[start:i + 1] = replacement_value
        
        i += 1
    
    return arr


def state_cps_function(pred_series):

    """
    Finds changepoints in a series based on changes in state.

    Args:
        pred_series (numpy array): The input series.

    Returns:
        list: The list of changepoints.
    """

    cps = []
    
    for i in range(1, len(pred_series)):
        if pred_series[i] != pred_series[i - 1] and i > 2 and i < len(pred_series) - 2:
            cps.append(i)

    # cps = [cp for cp in cps if cp > 2 and cp < len(pred_series)-2]
    cps.append(len(pred_series))

    return cps




# # get changepoints from all models
# def combined_cps(pred_series_alpha, pred_series_k):

#     """
#     Combines changepoints from multiple models.

#     Args:
#         pred_series_alpha (numpy array): The input series for the alpha model.
#         pred_series_k (numpy array): The input series for the k model.
#         pred_series_state (numpy array): The input series for the state model.

#     Returns:
#         tuple: A tuple containing the merged changepoints, alpha changepoints, k changepoints,
#                alpha series, k series, and state series.
#     """

#     pred_series_alpha = median_filter_1d(smooth_series(pred_series_alpha, lower_limit=0, upper_limit=1.999))
#     pred_series_k = median_filter_1d(smooth_series(pred_series_k, lower_limit=0, upper_limit=6))

#     alpha_cps = list(alpha_cps_function(pred_series_alpha, penalty=0.3, threshold=0.05))
#     k_cps = list(k_cps_function(pred_series_k, penalty=0.3, threshold=0.05))

#     combined_cps = list(set(k_cps + alpha_cps))
#     combined_cps.sort()

#     merged_cps = []
#     i = 0
    
#     while i < len(combined_cps):
#         current_cp = combined_cps[i]
#         merged_cps.append(current_cp)
        
#         # Skip all changepoints that are within 'threshold' steps of the current_cp
#         while i < len(combined_cps) and combined_cps[i] - current_cp < 3:
#             i += 1
    
#     merged_cps.sort()
    
#     return merged_cps, alpha_cps, k_cps, pred_series_alpha, pred_series_k






# # get changepoints from all models
# def combined_cps_with_state(pred_series_alpha, pred_series_k, pred_series_state):

#     """
#     Combines changepoints from multiple models.

#     Args:
#         pred_series_alpha (numpy array): The input series for the alpha model.
#         pred_series_k (numpy array): The input series for the k model.
#         pred_series_state (numpy array): The input series for the state model.

#     Returns:
#         tuple: A tuple containing the merged changepoints, alpha changepoints, k changepoints,
#                alpha series, k series, and state series.
#     """

#     pred_series_alpha = median_filter_1d(smooth_series(pred_series_alpha, lower_limit=0, upper_limit=1.999))
#     pred_series_k = median_filter_1d(smooth_series(pred_series_k, lower_limit=0, upper_limit=6))
#     pred_series_state = replace_short_sequences(pred_series_state, min_length=3)

#     alpha_cps = list(alpha_cps_function(pred_series_alpha, penalty=0.3, threshold=0.05))
#     k_cps = list(k_cps_function(pred_series_k, penalty=0.3, threshold=0.05))
#     state_cps = list(state_cps_function(pred_series_state))

#     combined_cps = list(set(k_cps + alpha_cps + state_cps))
#     combined_cps.sort()

#     merged_cps = []
#     i = 0
    
#     while i < len(combined_cps):
#         current_cp = combined_cps[i]
#         merged_cps.append(current_cp)
        
#         # Skip all changepoints that are within 'threshold' steps of the current_cp
#         while i < len(combined_cps) and combined_cps[i] - current_cp < 3:
#             i += 1
    
#     merged_cps.sort()
    
#     return merged_cps, alpha_cps, k_cps, pred_series_alpha, pred_series_k, pred_series_state






def combined_cps_k_focused(pred_series_alpha, pred_series_k, pred_series_state, window_size=5):
    """
    Combines changepoints from multiple models, prioritizing k changepoints.

    Args:
        pred_series_alpha (numpy array): The input series for the alpha model.
        pred_series_k (numpy array): The input series for the k model.
        pred_series_state (numpy array): The input series for the state model.

    Returns:
        tuple: A tuple containing the merged changepoints, alpha changepoints, k changepoints,
               alpha series, k series, and state series.
    """

    # Preprocess the series
    pred_series_alpha = median_filter_1d(smooth_series(pred_series_alpha, lower_limit=0, upper_limit=1.999))
    pred_series_k = median_filter_1d(smooth_series(pred_series_k, lower_limit=0, upper_limit=6))
    pred_series_state = replace_short_sequences(pred_series_state, min_length=3)

    # Get changepoints from each model
    alpha_cps = list(alpha_cps_function(pred_series_alpha, penalty=0.3, threshold=0.05))
    k_cps = list(k_cps_function(pred_series_k, penalty=0.3, threshold=0.05))

    # Initialize merged changepoints list
    merged_cps = []

    # First, add all k changepoints to merged list
    merged_cps.extend(k_cps)
    
    # Then, check alpha changepoints
    for alpha_cp in alpha_cps:
        # Check if this alpha changepoint is near any k changepoint
        near_k_cp = False
        for k_cp in k_cps:
            if abs(alpha_cp - k_cp) <= window_size:
                near_k_cp = True
                break
        
        # If not near any k changepoint, add the alpha changepoint
        if not near_k_cp:
            merged_cps.append(alpha_cp)

    # Sort the final list
    merged_cps.sort()

    # Remove any remaining points that are too close (as a final check)
    final_merged_cps = []
    i = 0
    while i < len(merged_cps):
        current_cp = merged_cps[i]
        final_merged_cps.append(current_cp)
        
        # Skip points within window_size
        while i < len(merged_cps) and merged_cps[i] - current_cp < 3:
            i += 1
        
        # if i < len(merged_cps) and i > 0:
        #     i += 1

    final_merged_cps.sort()
    
    return final_merged_cps, alpha_cps, k_cps, pred_series_alpha, pred_series_k, pred_series_state



def combined_cps_k_focused_with_state(pred_series_alpha, pred_series_k, pred_series_state, window_size=5):
    """
    Combines changepoints from multiple models, prioritizing k changepoints first, 
    then alpha, then state changepoints.

    Args:
        pred_series_alpha (numpy array): The input series for the alpha model.
        pred_series_k (numpy array): The input series for the k model.
        pred_series_state (numpy array): The input series for the state model.
        window_size (int): Size of window to check for nearby changepoints (default=5)

    Returns:
        tuple: A tuple containing the merged changepoints, alpha changepoints, k changepoints,
               alpha series, k series, and state series.
    """

    # Preprocess the series
    pred_series_alpha = median_filter_1d(smooth_series(pred_series_alpha, lower_limit=0, upper_limit=1.999))
    pred_series_k = median_filter_1d(smooth_series(pred_series_k, lower_limit=0, upper_limit=6))
    pred_series_state = replace_short_sequences(pred_series_state, min_length=3)

    # Get changepoints from each model
    alpha_cps = list(alpha_cps_function(pred_series_alpha, penalty=0.3, threshold=0.05))
    k_cps = list(k_cps_function(pred_series_k, penalty=0.3, threshold=0.05))
    state_cps = list(state_cps_function(pred_series_state))

    # Initialize merged changepoints list
    merged_cps = []

    # First, add all k changepoints to merged list
    merged_cps.extend(k_cps)
    
    # Then, check alpha changepoints
    for alpha_cp in alpha_cps:
        # Check if this alpha changepoint is near any k changepoint
        near_k_cp = False
        for k_cp in k_cps:
            if abs(alpha_cp - k_cp) <= window_size:
                near_k_cp = True
                break
        
        # If not near any k changepoint, add the alpha changepoint
        if not near_k_cp:
            merged_cps.append(alpha_cp)
            
    # Finally, check state changepoints
    for state_cp in state_cps:
        # Check if this state changepoint is near any existing changepoint
        near_existing_cp = False
        for existing_cp in merged_cps:
            if abs(state_cp - existing_cp) <= window_size:
                near_existing_cp = True
                break
        
        # If not near any existing changepoint, add the state changepoint
        if not near_existing_cp:
            merged_cps.append(state_cp)

    # Sort all changepoints
    merged_cps.sort()

    # Remove any remaining points that are too close
    final_merged_cps = []
    i = 0
    while i < len(merged_cps):
        current_cp = merged_cps[i]
        final_merged_cps.append(current_cp)
        
        # Skip points within window_size
        while i < len(merged_cps) and merged_cps[i] - current_cp < 3:
            i += 1

    final_merged_cps.sort()
    
    return final_merged_cps, alpha_cps, k_cps, pred_series_alpha, pred_series_k, pred_series_state