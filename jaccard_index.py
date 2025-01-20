import os
from tqdm import tqdm
import numpy as np
from utils.postprocessing import smooth_series, median_filter_1d
import ruptures as rpt
from andi_datasets.utils_challenge import *
import json
import matplotlib.pyplot as plt  

# def cp_ground_truth(array):
#     cps = [0]
#     for i in range(1, len(array)):
#         if array[i-1] != array[i]:
#             cps.append(i)
#     return cps + [len(array)]

def getCP_rpt(array, lower_limit=0, upper_limit=float("inf"), threshold=0.05):
    array = median_filter_1d(smooth_series(array, lower_limit=lower_limit, upper_limit=upper_limit))
    if np.max(array) != np.min(array):
        pred_series_scaled = (array - np.min(array)) / (np.max(array) - np.min(array))
    else:
        pred_series_scaled = np.ones(len(array)) * 0.5 #scale them to default value of 0.5

    algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(pred_series_scaled)
    cps = [0] + algo.predict(pen=0.3)

    remove = []
    for i in range(1, len(cps) - 1):
        left_mean = array[cps[i - 1]:cps[i]].mean()
        right_mean = array[cps[i]:cps[i + 1]].mean()        
        if abs(left_mean - right_mean) < threshold:
            remove.append(cps[i])
    
    cps = [cp for cp in cps if cp not in remove]

    return cps, array


if __name__ == "__main__":

    # ROOT_DIR = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/jaccard_simulations_for_alpha_fixedL_singleCP_results"
    ROOT_DIR = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/K_jaccard_single_CP_more_sampling_results"
    print(os.path.basename(ROOT_DIR))
    predictions = np.load(os.path.join(ROOT_DIR, "pred_k.npy"))
    ground_truth = np.load(os.path.join(ROOT_DIR, "gt_k.npy"))

    bins = np.arange(-0.5, 0.55, 0.025)  # 2.25 to include 2
    counter_dict = {f"{(bins[i] + bins[i+1])/2:.2f}": 0 for i in range(len(bins)-1)}
    MAX_SAMPLES_PER_BIN = 50

    # NUMBER_OF_SAMPLES = 2000
    # progress_bar = tqdm(total=NUMBER_OF_SAMPLES, desc="Jaccard Index")

    d = {}
    rm = {}
    n0 = {}
    cp_gt = [100]
    x = 0
    # target_deltas = [str(round(x * 0.05, 2)) for x in range(1, 40)]  # Creates [0.05, 0.10, ..., 1.95]
    # counter_dict = {}

    # from collections import Counter

    # counter_dict = Counter()

    # for i in range(len(ground_truth)):
    #     progress_bar.update()
        
    #     g = ground_truth[i]
    #     delta = round(g[0] - g[-1], 2)
        
    #     counter_dict[delta] += 1

    # # # After the loop, counter_dict will contain the count of each delta value
    # print(counter_dict)
    # # exit()

    for i in range(len(ground_truth)):
        # i =4000
        # time = [i for i in range(len(ground_truth[i]))]

        # p = predictions[i]
        # g = ground_truth[i]

        # cpt, p = getCP_rpt(p)
        # rmse, jaccard_value = single_changepoint_error([100], cpt[1:-1])
        # print(jaccard_value)
        # print(cpt[1:-1])
        # plt.scatter(time, p)
        # plt.scatter(time, g)
        # plt.show()

        # exit()

        # progress_bar.update()
        g = ground_truth[i]
        delta = round(g[0] - g[-1], 2)
            # Find which bin this delta belongs to
        for bin_start, bin_end in zip(bins[:-1], bins[1:]):
            if bin_start <= delta < bin_end:
                bin_center = f"{(bin_start + bin_end)/2:.2f}"
                # Skip if this bin is already full
                if counter_dict[bin_center] >= MAX_SAMPLES_PER_BIN:
                    break
                
                # Process the sample
                p = predictions[i]
                cp_pred, _ = getCP_rpt(p, lower_limit=0, upper_limit=6)
                cp_pred = cp_pred[1:-1]
                rmse, jaccard_value = single_changepoint_error(cp_gt, cp_pred)
                
                # Update your dictionaries
                if bin_center not in d:
                    d[bin_center] = jaccard_value
                else:
                    d[bin_center] += jaccard_value
                
                counter_dict[bin_center] += 1
                break

        total_samples = sum(counter_dict.values())
        progress_percent = (total_samples / (40 * 50)) * 100
        print(f"Progress: {progress_percent:.1f}%", end="\r")
            
        # Check if all bins have reached the maximum
        if all(count >= MAX_SAMPLES_PER_BIN for count in counter_dict.values()):
            break
    # for i in range(len(ground_truth)):

    #     progress_bar.update()

    #     p = predictions[i]
    #     g = ground_truth[i]

    #     # delta = str(round((10**g[0] - 1)/(10**g[100]-1),2))
    #     # delta = str(round((10**g[0]-1) - (10**g[-1]-1),2))
    #     delta = str(round(g[0] - g[-1], 2))

    #     cp_pred, _ = getCP_rpt(p, lower_limit=0, upper_limit=6)
    #     cp_pred = cp_pred[1:-1]

    #     rmse, jaccard_value = single_changepoint_error(cp_gt, cp_pred)

    #     if delta not in d:
    #         d[delta] = jaccard_value
    #         counter_dict[delta] = 1
    #         # rm[delta] = rmse
    #         # n0[delta] = len(cp_pred)
    #     else:
    #         d[delta] += jaccard_value
    #         counter_dict[delta] += 1
    #         # rm[delta] += rmse
    #         # n0[delta] += len(cp_pred)

    #         # print(jaccard_value)


    with open(os.path.join(ROOT_DIR, "jaccard_more_sampling_new_tracks_0_5.json"), "w") as file:
        json.dump(d, file, indent=4)
    with open(os.path.join(ROOT_DIR, "counter_more_sampling_0_5.json"), "w") as file:
        json.dump(counter_dict, file, indent=4)
    # with open(os.path.join(ROOT_DIR, "number_more_sampling_.json"), "w") as file:
    #     json.dump(n0, file, indent=4)
    # with open(os.path.join(ROOT_DIR, "rmse_more_sampling_.json"), "w") as file:
    #     json.dump(rm, file, indent=4)
    print("Done Jaccard Indices")