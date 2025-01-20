import os
os.environ["TQDM_DISABLE"] = "1"
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Lock, Manager
import contextlib
import io
import sys 
import collections
import random 

def generate_sample(k1, k2):
    state = 1 # multi
    dic = _get_dic_andi2(state + 1)    
    dic['T'] = 200 
    dic['N'] = 100
    dic['alphas'] = np.array([[1, 0], [1, 0]])
    k1 = np.round(10**k1 - 1,2)
    k2 = np.round(10**k2 - 1,2)
    dic['Ds'] = np.array([[k1, 0],[k2, 0]])
    return dic

def process_alpha(args):

    delta_k_pair, counter, lock = args
    tracks_generated = 0
    delta_k, k1, k2 = delta_k_pair

    # attempts = 0

    while tracks_generated < 100:

        dic = generate_sample(k1, k2)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
                                                        dics=[dic], 
                                                        return_timestep_labs=True, 
                                                        get_video=False)

            # if attempts == 3:
            #     break

        except:

            # attempts += 1
            # if attempts == 3:
            #     break
            continue

        if dfs_traj != 0:
            for _, group in dfs_traj[0].groupby('traj_idx'):
                if int(len(group)) == 200:
                    group.to_parquet(os.path.join(SAVE_DIR, f"delta_K_{delta_k:.2f}_track_{tracks_generated}.parquet"))
                    tracks_generated += 1
                    if tracks_generated >= 100:
                        break
                    # attempts = 0
        # else:
        #     attempts += 1

        # if attempts == 3:
        #     break
    
    with lock:
        counter.value += 1
        print(f"\rProcessed: {counter.value} out of {TOTAL_POSSIBILITIES}", end="\r")
    
    # print(f"\rProcessed: {delta_k} \n")
    return delta_k, tracks_generated

if __name__ == "__main__":
    NUM_WORKERS = 30
    SAVE_DIR = "K_jaccard_single_CP_more_sampling_finer"

    # k1_list = [round(i*0.1,2) for i in range(1,201)]
    # k2_list = [round(0.05 * i, 2) for i in range(1, 42)]
    # k2_list = [round(i*0.1,2) for i in range(1,201)]
    # print(k1_list)
    log1p_k1_value_list = [round(0.01 * i, 2) for i in range(0, 201, 1)][1:]
    log1p_k2_value_list = [round(0.01 * i, 2) for i in range(0, 201, 1)][1:]

    # dic = _get_dic_andi2(1 + 1)    
    # dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
    #                                 dics=[dic], 
    #                                 return_timestep_labs=True, 
    #                                 get_video=False)
    # df = dfs_traj[0]
    # df = df[df["traj_idx"] ==0]
    # ar = np.round(np.array(df["D"]),2)
    # counter = 0
    # for i in range(1, len(ar)):
    #     if ar[i] != ar[i-1]:
    #         counter+=1
    # print(counter)
    # exit()


    l = []
    delta_values = set()

    for k1 in log1p_k1_value_list:
        for k2 in log1p_k2_value_list:
            delta = round(k1 - k2, 2)
            t = (delta, k1, k2)
            if delta not in delta_values:
                delta_values.add(delta)
                l.append(t)

    TOTAL_POSSIBILITIES = len(l)

    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    with Pool(NUM_WORKERS) as p:
        # results = p.map(process_alpha, delta_alpha_items)
        results = p.map(process_alpha, [(item, counter, lock) for item in l])

    # for alpha_pairs, tracks in results:
    #     print(f"Alpha: {alpha_pairs:.2f}, Tracks generated: {tracks}")
    print("\nAll K values processed.")