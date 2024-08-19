import numpy as np
import ruptures as rpt

# changepoint is the first index of the new segment

def smooth_series(series, lower_limit=None, upper_limit=None):

    if lower_limit is not None:
        series[series < lower_limit] = lower_limit
    
    if upper_limit is not None:
        series[series > upper_limit] = upper_limit

    smoothed_series = []
    moving_avg = []

    for i in range(len(series) - 1):
        if abs(series[i+1] - series[i]) < 0.01:
            moving_avg.append(series[i])
        else:
            if moving_avg:
                moving_avg.append(series[i])
                average = np.mean(moving_avg)
                smoothed_series.extend([average] * len(moving_avg))
                moving_avg = []
            else:
                smoothed_series.append(series[i])
    
    # Handle the last element or the remaining moving average
    if moving_avg:
        moving_avg.append(series[-1])
        average = np.mean(moving_avg)
        smoothed_series.extend([average] * len(moving_avg))
    else:
        smoothed_series.append(series[-1])

    return smoothed_series

def median_filter_1d(arr):

    filtered_arr = np.zeros_like(arr)
    padded_arr = np.pad(arr, pad_width=1, mode='edge')
    
    # Apply the median filter with a window size of 3
    for i in range(len(arr)):
        window = padded_arr[i:i+3]
        filtered_arr[i] = np.median(window)
    
    filtered_arr[0] = filtered_arr[2]
    filtered_arr[1] = filtered_arr[2]

    filtered_arr[-1] = filtered_arr[-3]
    filtered_arr[-2] = filtered_arr[-3]

    return filtered_arr


def alpha_cps_function(pred_series,penalty):
    
    if np.max(pred_series) != np.min(pred_series):
        pred_series_scaled = (pred_series - np.min(pred_series)) / (np.max(pred_series) - np.min(pred_series))
    else:
        pred_series_scaled = pred_series

    algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(pred_series_scaled)
    cps = algo.predict(pen=penalty)
    
    cps = [cp for cp in cps if cp > 2 and cp < len(pred_series)-2]
    cps.append(len(pred_series))

    cps = [0] + cps

    remove = []
    for i in range(1, len(cps) - 1):
        left = pred_series[cps[i - 1]:cps[i]]
        right = pred_series[cps[i]:cps[i + 1]]
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        
        if abs(left_mean - right_mean) < 0.1:
            remove.append(i)
    
    cps = [cp for i, cp in enumerate(cps) if i not in remove]
    cps.pop(0)

    return cps

def k_cps_function(pred_series, penalty):
    
    if np.max(pred_series) != np.min(pred_series):
        pred_series_scaled = (pred_series - np.min(pred_series)) / (np.max(pred_series) - np.min(pred_series))
    else:
        pred_series_scaled = pred_series

    algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(pred_series_scaled)
    cps = algo.predict(pen=penalty)
    
    cps = [cp for cp in cps if cp > 2 and cp < len(pred_series)-2]
    cps.append(len(pred_series))

    cps = [0] + cps

    remove = []
    for i in range(1, len(cps) - 1):
        left = pred_series[cps[i - 1]:cps[i]]
        right = pred_series[cps[i]:cps[i + 1]]
        left_mean = np.mean(left)
        right_mean = np.mean(right)
        if abs(left_mean - right_mean) < 0.05:
            remove.append(i)
    
    cps = [cp for i, cp in enumerate(cps) if i not in remove]
    cps.pop(0)

    return cps


def replace_short_sequences(arr, min_length=3):
    arr = np.array(arr)  # Ensure input is a numpy array for easy manipulation
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
    
    cps = []
    for i in range(1, len(pred_series)):
        if pred_series[i] != pred_series[i - 1]:
            cps.append(i)

    cps = [cp for cp in cps if cp > 2 and cp < len(pred_series)-2]
    cps.append(len(pred_series))

    return cps

# get changepoints from all models
def combined_cps(pred_series_alpha, pred_series_k, pred_series_state):

    pred_series_alpha = median_filter_1d(smooth_series(pred_series_alpha, lower_limit=0, upper_limit=1.999))
    pred_series_k = median_filter_1d(smooth_series(pred_series_k, lower_limit=0, upper_limit=6))
    pred_series_state = replace_short_sequences(pred_series_state, min_length=3)

    alpha_cps = list(alpha_cps_function(pred_series_alpha, penalty=0.3))
    k_cps = list(k_cps_function(pred_series_k, penalty=0.3))

    combined_cps = list(set(k_cps + alpha_cps))
    combined_cps.sort()

    merged_cps = []
    i = 0
    
    while i < len(combined_cps):
        current_cp = combined_cps[i]
        merged_cps.append(current_cp)
        
        # Skip all changepoints that are within 'threshold' steps of the current_cp
        while i < len(combined_cps) and combined_cps[i] - current_cp < 3:
            i += 1
    
    merged_cps.sort()
    
    return merged_cps, alpha_cps, k_cps, pred_series_alpha, pred_series_k, pred_series_state