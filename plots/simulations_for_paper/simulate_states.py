import os
os.environ["TQDM_DISABLE"] = "1" # Suppresses the tqdm progress bar from challenge_phenom_dataset function for clearer output
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Manager
import random 
import contextlib
import io
import numpy as np

def process_states(item, counters, lock):
    tracks_generated = 0
    dic, state_filter = item[0], item[1]
    length = dic["T"]

    while tracks_generated < 100:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
                                                        dics=[dic], 
                                                        return_timestep_labs=True, 
                                                        get_video=False)
        except:
            continue

        if dfs_traj != 0:
            for _, group in dfs_traj[0].groupby('traj_idx'):
                if int(len(group)) == length and (group['state'] == state_filter).all():
                    group.to_parquet(os.path.join(SAVE_DIR, f"state{state_filter}_length_{length:.2f}_track_{tracks_generated}.parquet"))
                    tracks_generated += 1
                    with lock:
                        counters[str(state_filter)].value += 1
                        counter_dict = {str(k): v.value for k, v in counters.items()}
                        print(f"\rCounters: {counter_dict} | Length: {length}", end="")
                    if tracks_generated >= 100:
                        break

    return length, tracks_generated



if __name__ == "__main__":

    SAVE_DIR = "fixed_states_over_length"
    NUM_WORKERS = 30

    os.makedirs(SAVE_DIR, exist_ok=True)

    length_list = [i for i in range(20,201)]
    dics = []

    # directed s = 3
    for l in length_list:
        dic = _get_dic_andi2(1)
        dic['T'] = l 
        dic['N'] = 100
        dic['alphas'] = np.array([1.95, 0])
        dic['Ds'] = np.array([1, 0])  

        dics.append([dic, 3])

    # # free s = 2
    for l in length_list:
        dic = _get_dic_andi2(1)
        dic['T'] = l 
        dic['N'] = 100
        dic['alphas'] = np.array([1, 0])
        dic['Ds'] = np.array([1, 0])  

        dics.append([dic, 2])

    # # immobile s=0
    for l in length_list:
        dic = _get_dic_andi2(3)
        dic['T'] = l
        dic['Pu'] = 0
        dic['Pb'] = 1
        dic['Nt'] = 600
        dic["r"] = 1

        dics.append([dic, 0])

    # # confined s=1
    for l in length_list:
        dic = _get_dic_andi2(5)

        dic['T'] = l 
        Ds_array = np.array([[1, 0],[1, 0]])
        alphas_array = np.array([[1, 0],[1, 0]])

        dic['Ds'] = Ds_array
        dic['alphas'] = alphas_array
        dic['trans'] = 0
        dic['Nc'] = 100
        dic["r"] = 7.5

        dics.append([dic, 1])

    print("Done generating dics", len(dics))

    manager = Manager()
    counters = {
        '0': manager.Value('i', 0),
        '1': manager.Value('i', 0),
        '2': manager.Value('i', 0),
        '3': manager.Value('i', 0)
    }

    lock = manager.Lock()

    with Pool(30) as p:
        p.starmap(process_states, [(item, counters, lock) for item in dics])

  