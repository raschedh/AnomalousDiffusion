import os 
import numpy as np
import matplotlib.pyplot as plt


ROOT_DIR = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/test_set"
LABEL_PADDING_VALUE = 99

for file in os.listdir(ROOT_DIR):
    path = os.path.join(ROOT_DIR, file)

    if file == "gt_a.npy":
        gt_a = np.load(path)
    
    if file == "gt_k.npy":
        gt_k = np.load(path)
    
    if file == "gt_state.npy":
        gt_state = np.load(path)


def getCP_gt(array):
    cps = [0]
    for i in range(1, len(array)):
        if array[i-1] != array[i]:
            cps.append(i)

    return cps + [len(array)]

def padding_starts_index(array):
    padding_starts = (array == LABEL_PADDING_VALUE).argmax() 
    if padding_starts == 0:
        padding_starts = 200 
    return padding_starts


total_cps = 0
special_cases = 0
no_of_tracks_with_this = 0
total_tracks = len(gt_a)

for i in range(len(gt_a)):    
    print(round(i/len(gt_a) *100, 0), end="\r")
    idx_a = padding_starts_index(gt_a[i])
    g_alpha = gt_a[i][:idx_a]
    g_k = gt_k[i][:idx_a]
    g_state = gt_state[i][:idx_a]
    
    cp_a = len(getCP_gt(g_alpha))
    cp_k = len(getCP_gt(g_k))
    cp_s = len(getCP_gt(g_state))
    
    if (cp_s) > (cp_a) and (cp_s) > (cp_k):
        special_cases += (cp_s) - max((cp_a), (cp_k))
        no_of_tracks_with_this += 1
    
    total_cps += max(cp_a, cp_k, cp_s)
    # no_of_cps = len(cp_a) - 2
    # if no_of_cps <=5:
    #     og = getCP_gt(g_alpha)
    #     state_only_cps = cp_s - (cp_a | cp_k)

    #     if state_only_cps:  
    #         # Create figure with 3 subplots
    #         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
    #         # Time array
    #         t = np.arange(len(g_alpha))
            
    #         # Plot alpha
    #         ax1.plot(t, g_alpha, 'b-', label='α')
    #         for cp in cp_a:
    #             ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    #         ax1.set_ylabel('α')
    #         ax1.grid(True)
            
    #         # Plot K
    #         ax2.plot(t, g_k, 'g-', label='K')
    #         for cp in cp_k:
    #             ax2.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    #         ax2.set_ylabel('K')
    #         ax2.grid(True)
            
    #         # Plot state
    #         ax3.plot(t, g_state, 'm-', label='State')
    #         for cp in cp_s:
    #             ax3.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    #         ax3.set_ylabel('State')
    #         ax3.set_xlabel('Time')
    #         ax3.grid(True)
            
    #         plt.tight_layout()
    #         plt.show()
            
    #         print(f"Changepoints - Alpha: {cp_a}, K: {cp_k}, State: {cp_s}")
    #         print(f"State-only changepoints: {state_only_cps}")
    #         plt.show()
    #     total_cps += len(cp_k) 

print("special tracks", no_of_tracks_with_this, "out of", total_tracks)
print("cp places", special_cases, " out of cps", total_cps)
