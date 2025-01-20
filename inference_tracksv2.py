import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from utils.pad_batch import pad_batch, LABEL_PADDING_VALUE
from models.RegressionModel import RegressionModel
from models.ClassificationModel import ClassificationModel
import pickle 
from pathlib import Path
import numpy as np
from utils.timeseriesdataset import TimeSeriesDataset
import os 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LENGTH = 200
PADDING_VALUE = 99

# AlphaModel = RegressionModel().to(DEVICE)
KModel = RegressionModel().to(DEVICE)
# StateModel = ClassificationModel().to(DEVICE)

# AlphaModel.load_state_dict(torch.load("models/optimal_weights/alpha_weights"))
# AlphaModel.load_state_dict(torch.load("/home/haidiri/Desktop/AnDiChallenge2024/models/optimal_weights/alpha_weights_with_fixed"))
KModel.load_state_dict(torch.load("models/optimal_weights/k_weights"))
# StateModel.load_state_dict(torch.load("/home/haidiri/Desktop/AnDiChallenge2024/models/checkpoints/state_model/model_20241001_115850_weighed_on_val/model_25_23"))

# AlphaModel.eval()
KModel.eval()
# StateModel.eval()

def pad_tensor(tensor, target_length=200, pad_value=99):
    """Pad the last dimension of a tensor to the target length."""
    current_length = tensor.shape[-1]
    if current_length >= target_length:
        return tensor[..., :target_length]
    padding = (0, target_length - current_length)
    return torch.nn.functional.pad(tensor, padding, mode='constant', value=pad_value)


if __name__ == "__main__":
    # PICKLE_FILE = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/jaccard_simulations_for_alpha_fixedL_threeCP_results/jaccard_simulations_for_alpha_fixedL_threeCP.pkl"
    # PICKLE_FILE = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/single_state_k_values_all_point_based_results/single_state_k_values_all_point_based.pkl"
    # PICKLE_FILE = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/jaccard_simulations_for_alpha_fixedL_singleCP_results/jaccard_simulations_for_alpha_fixedL_singleCP.pkl"
    # PICKLE_FILE = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/K_jaccard_single_CP_results/K_jaccard_single_CP.pkl"
    # PICKLE_FILE = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/fixed_states_over_length_results/fixed_states_over_length.pkl"
    PICKLE_FILE = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/K_jaccard_single_CP_more_sampling_results/K_jaccard_single_CP_more_sampling.pkl"

    with open(PICKLE_FILE, "rb") as f:
        data = pickle.load(f)

    # with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_general/test_instances.pkl", "rb") as file:
    #     data = pickle.load(file)

    # with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_3bias/test_instances.pkl", "rb") as file:
    #     data.extend(pickle.load(file))

    # with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_default/test_instances.pkl", "rb") as file:
    #     data.extend(pickle.load(file))

    # with open("/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulation_alpha_K_one_fixed/test_instances.pkl", "rb") as file:
    #     data.extend(pickle.load(file))

    SAVE_DIR, _ = os.path.split(PICKLE_FILE)
    # SAVE_DIR = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/test_set"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Saving to dir:", SAVE_DIR)

    concat_data = ConcatDataset(data)
    loader = DataLoader(concat_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_batch)
    print("Test data: ", len(data), "DataLoader Sizes:", len(loader))

    # Initialize empty lists
    pred_a_list, pred_k_list, pred_state_list = [], [], []
    gt_a_list, gt_k_list, gt_state_list = [], [], []

    progress_bar = tqdm(total=len(loader), desc='Testing', position=0)

    with torch.no_grad():
        for inputs, alpha_labels, k_labels, state_labels in loader:
            inputs = inputs.to(DEVICE)
            
            # pred_alpha_per_track = AlphaModel(inputs).squeeze(-1)
            pred_k_per_track = KModel(inputs).squeeze(-1)
            # pred_state_per_track = torch.argmax(StateModel(inputs), dim=-1)

            # # Pad predictions
            # pred_alpha_per_track = pad_tensor(pred_alpha_per_track, target_length=200, pad_value=99)
            # pred_k_per_track = pad_tensor(pred_k_per_track, target_length=200, pad_value=99)
            # pred_state_per_track = pad_tensor(pred_state_per_track, target_length=200, pad_value=99)

            # Append padded predictions to lists
            # pred_a_list.append(pred_alpha_per_track)
            pred_k_list.append(pred_k_per_track)
            # pred_state_list.append(pred_state_per_track)

            # gt_a_list.append(alpha_labels)
            gt_k_list.append(k_labels)
            # gt_state_list.append(state_labels)

            # # Pad labels
            # gt_a_list.append(pad_tensor(alpha_labels, target_length=200, pad_value=99))
            # gt_k_list.append(pad_tensor(k_labels, target_length=200, pad_value=99))
            # gt_state_list.append(pad_tensor(state_labels, target_length=200, pad_value=99))

            progress_bar.update()

    progress_bar.close()

    # Convert lists to tensors
    # pred_a = torch.cat(pred_a_list, dim=0)
    pred_k = torch.cat(pred_k_list, dim=0)
    # pred_state = torch.cat(pred_state_list, dim=0)
    # gt_a = torch.cat(gt_a_list, dim=0)
    gt_k = torch.cat(gt_k_list, dim=0)
    # gt_state = torch.cat(gt_state_list, dim=0)

    # print("Predictions shape:", pred_state.shape)
    # print("Ground truth shape:", gt_state.shape)

    print("Predictions shape:", pred_k.shape)
    print("Ground truth shape:", gt_k.shape)

    # Convert to numpy and save
    print("Converting to numpy and saving...")
    # np.save(os.path.join(SAVE_DIR, "pred_a.npy"), pred_a.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "pred_k.npy"), pred_k.cpu().numpy())
    # np.save(os.path.join(SAVE_DIR, "pred_state.npy"), pred_state.cpu().numpy())
    # np.save(os.path.join(SAVE_DIR, "gt_a.npy"), gt_a.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "gt_k.npy"), gt_k.cpu().numpy())
    # np.save(os.path.join(SAVE_DIR, "gt_state.npy"), gt_state.cpu().numpy())

    print("Processing completed successfully.")