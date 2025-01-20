import pickle 
import random 

def load_data():

    # train_data = []
    # val_data = []

    #########################################################################################################
    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_general/train_instances.pkl", "rb") as file:
        train_data = pickle.load(file)

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_3bias/train_instances.pkl", "rb") as file:
        train_data.extend(pickle.load(file))

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_default/train_instances.pkl", "rb") as file:
        train_data.extend(pickle.load(file))

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulation_alpha_K_one_fixed/train_instances.pkl", "rb") as file:
        train_data.extend(pickle.load(file))

    ##########################################################################################################

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_general/val_instances.pkl", "rb") as file:
        val_data = pickle.load(file)

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_3bias/val_instances.pkl", "rb") as file:
        val_data.extend(pickle.load(file))

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_default/val_instances.pkl", "rb") as file:
        val_data.extend(pickle.load(file))

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulation_alpha_K_one_fixed/val_instances.pkl", "rb") as file:
        val_data.extend(pickle.load(file))


    ##########################################################################################################

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_general/test_instances.pkl", "rb") as file:
        test_data = pickle.load(file)

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_3bias/test_instances.pkl", "rb") as file:
        test_data.extend(pickle.load(file))

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_default/test_instances.pkl", "rb") as file:
        test_data.extend(pickle.load(file))

    with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulation_alpha_K_one_fixed/test_instances.pkl", "rb") as file:
        test_data.extend(pickle.load(file))


    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    move_to_val = 0.2 * len(train_data)

    val_data.extend(train_data[:int(move_to_val)])
    train_data = train_data[int(move_to_val):]

    return train_data, val_data, test_data
























