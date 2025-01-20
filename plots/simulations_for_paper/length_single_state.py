import os
os.environ["TQDM_DISABLE"] = "1"
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool
import contextlib
import io

def generate_sample(length):

    state = 0
    dic = _get_dic_andi2(state + 1)    
    dic['T'] = length 
    dic['N'] = 100
    dic['alphas'], dic['Ds'] = np.array([1, 0]), np.array([1, 0])
    return dic

def process_k(length):
    tracks_generated = 0
    
    while tracks_generated < 1000:
        dic = generate_sample(length=length)
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
                if int(len(group)) == length:
                    group.to_parquet(os.path.join(SAVE_DIR, f"k_{length:.2f}_track_{tracks_generated}.parquet"))
                    tracks_generated += 1
                    if tracks_generated >= 1000:
                        break
    
    print(f"\nCompleted = {length:.2f}, Total tracks: {tracks_generated}")
    return length, tracks_generated

if __name__ == "__main__":
    NUM_WORKERS = 30
    length_list = [i for i in range(20,201)]
    SAVE_DIR = "single_state_length_values"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Processing {len(length_list)} values using {NUM_WORKERS} workers")
    
    with Pool(NUM_WORKERS) as p:
        results = p.map(process_k, length_list)

    for length, tracks in results:
        print(f"k: {length:.2f}, Tracks generated: {tracks}")

    print("\nAll values processed.")