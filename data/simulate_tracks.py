from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Manager
import random 
import os

# for faster simulated tracks, we use multiprocessing and 
# we save each individiual track as parquet file for faster reduced size and in case the simulation fails at any point. This file uses the
# ANDI github library and simulated proteins in states: single, multi, dimer, immobile, confined 

def sample_alpha_K(range_alpha, range_K):
    
    log_K_plus1, alpha = np.random.choice(range_K), np.random.choice(range_alpha)
    K = np.round(10**log_K_plus1 - 1, 2)
    return alpha, K

def generate_random_transition_matrix(n):
    transition_matrix = np.zeros((n, n))

    # Set diagonal values between 0.5 and 0.9
    for i in range(n):
        transition_matrix[i, i] = np.random.uniform(0.5, 0.9)
    
    # Fill the off-diagonal values to ensure each row sums to 1
    for i in range(n):
        remaining_probability = 1 - transition_matrix[i, i]
        off_diagonal_values = np.random.dirichlet(np.ones(n - 1)) * remaining_probability
        transition_matrix[i, :i] = off_diagonal_values[:i]
        transition_matrix[i, i + 1:] = off_diagonal_values[i:]
    
    return transition_matrix

# Worker function for multiprocessing
def process_dic(dic, counters, lock, limit=100000):

    model_type = dic['model']
    SAVE_PATH = os.path.join(SAVE_DIR, model_type)

    with lock:
        if counters[model_type].value >= limit:
            return
        
    try:
        dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
                                                  dics=[dic], 
                                                  return_timestep_labs=True, 
                                                  get_video=False)
    except:
        return

    if dfs_traj != 0:
        for _, group in dfs_traj[0].groupby('traj_idx'):
            with lock:
                if counters[model_type].value >= limit:
                    return
                counters[model_type].value += 1
                counter_values = {key: counters[key].value for key in counters}
                print(f"Counters: {counter_values}", end='\r')
                group.to_parquet(os.path.join(SAVE_PATH, f"track_{counters[model_type].value}.parquet"))


if __name__ == "__main__":

    SAVE_DIR = "./simulated_tracks"
    os.makedirs(SAVE_DIR, exist_ok=True)

    MODELS = np.arange(5)
    REPEATS = 400000
    NUM_WORKERS = 30

    range_k = np.linspace(0, 6, 601)[1:] #K range from 10e-12 to 10e6 which we transform to log10(K + 1) becoming [0,6]
    range_alpha = np.linspace(0, 2, 201)[1:-1] #alpha ranges from [0,2]
    # range_alpha3 = range_alpha[range_alpha_linspace >=1.9] #if we want to simulate more proteins with state 3 change to this

    print("Start K:", range_k[0], "End K:", range_k[-1])
    print("Start Alpha:",range_alpha[0], "End Alpha:", range_alpha[-1])

    dics = []

    for m in MODELS:        
        for _ in range(REPEATS):  

            dic = _get_dic_andi2(m+1)
            dic['T'] = 200 
            dic['N'] = 100

            ### SINGLE STATE ####
            if m == 0:  

                # sample alpha and K 
                alpha, K = sample_alpha_K(range_alpha, range_k)

                dic['alphas'] = np.array([alpha, 0.1*alpha])
                dic['Ds'] = np.array([K, 0.1*K])
                
                os.makedirs(os.path.join(SAVE_DIR, dic['model']), exist_ok=True)

            ### MULTI STATE ####
            if m == 1:

                base_states = 2 
                if np.random.rand() < 0.5:
                    base_states += 1
                
                # generate transition matrix 
                D_array = np.zeros((base_states, 2))
                alpha_array = np.zeros((base_states, 2))

                for i in range(base_states):
                    alpha, K = sample_alpha_K(range_alpha, range_k)                    
                    D_array[i] = np.array([K, 0.1*K])
                    alpha_array[i] = np.array([alpha, 0.01*alpha])

                dic['M'] = generate_random_transition_matrix(base_states)
                dic['Ds'] = D_array
                dic['alphas'] = alpha_array

                os.makedirs(os.path.join(SAVE_DIR, dic['model']), exist_ok=True)

            #### IMMOBILE TRAPS ####
            if m == 2:
                alpha, K = sample_alpha_K(range_alpha, range_k)

                dic['Ds'] = np.array([K, 0.1*K])
                dic['alphas'] = np.array([alpha, 0.01*alpha])
                dic['Pu'] = random.uniform(0, 0.1)
                dic['Pb'] = 1
                dic['r'] = random.uniform(0.5, 1)
                dic['Nt'] = random.randint(100, 300)

                os.makedirs(os.path.join(SAVE_DIR, dic['model']), exist_ok=True)
            
            ### DIMERIZATION ####
            if m == 3:
                
                Ds_array = np.zeros((2, 2))
                alphas_array = np.zeros((2, 2))
                
                alpha_1, K_1 = sample_alpha_K(range_alpha, range_k)

                Ds_array[0] = np.array([K_1, 0.1*K_1])
                alphas_array[0] = np.array([alpha_1, 0.01*alpha_1])

                alpha_2, K_2 = sample_alpha_K(range_alpha, range_k)

                Ds_array[1] = np.array([K_2, 0.1*K_2])
                alphas_array[1] = np.array([alpha_2, 0.01*alpha_2])

                dic['Pu'] = random.uniform(0, 0.1)
                dic['Pb'] = 1
                dic['r'] = random.uniform(0.5, 1)

                os.makedirs(os.path.join(SAVE_DIR, dic['model']), exist_ok=True)
                
            ### CONFINEMENT ####
            if m == 4:

                Ds_array = np.zeros((2, 2))
                alphas_array = np.zeros((2, 2))

                alpha_1, K_1 = sample_alpha_K(range_alpha, range_k)

                Ds_array[0] = np.array([K_1, 0.1*K_1])
                alphas_array[0] = np.array([alpha_1, 0.01*alpha_1])

                alpha_confined, K_confined = sample_alpha_K(range_alpha, range_k)

                Ds_array[1] = np.array([K_confined, 0.1*K_confined])
                alphas_array[1] = np.array([alpha_confined, 0.1*alpha_confined])

                dic['trans'] = random.uniform(0, 0.3)
                dic['Nc'] = random.randint(30,50)
                dic['r'] = random.randint(5,12)

                os.makedirs(os.path.join(SAVE_DIR, dic['model']), exist_ok=True)

            dics.append(dic)


    print("Done generating dics", len(dics))
    random.shuffle(dics)

    # save it to correct folders 
    manager = Manager()
    counters = {
        'single_state': manager.Value('i', 0),
        'multi_state': manager.Value('i', 0),
        'immobile_traps': manager.Value('i', 0),
        'dimerization': manager.Value('i', 0),
        'confinement': manager.Value('i', 0)
    }

    lock = manager.Lock()

    with Pool(NUM_WORKERS) as p:
        p.starmap(process_dic, [(dic, counters, lock) for dic in dics])

  