from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

LABEL_PADDING_VALUE = 99
FEATURE_PADDING_VALUE = 0

def pad_batch(batch):

    """
    Pads a batch of sequences.

    Args:
        batch (list): A list of tuples containing the features, alpha labels, k labels, and state labels.

    Returns:
        tuple: A tuple containing the padded features, alpha labels, k labels, and state labels.
    """

    features, alpha_labels, k_labels, state_labels = zip(*batch)    
    features_padded = pad_sequence(features, batch_first=True, padding_value=FEATURE_PADDING_VALUE)
    alpha_labels_padded = pad_sequence(alpha_labels, batch_first=True, padding_value=LABEL_PADDING_VALUE)
    k_labels_padded = pad_sequence(k_labels, batch_first=True, padding_value=LABEL_PADDING_VALUE)
    state_labels_padded = pad_sequence(state_labels, batch_first=True, padding_value=LABEL_PADDING_VALUE)

    return features_padded, alpha_labels_padded, k_labels_padded, state_labels_padded


def pad_tensor(tensor, target_length=200, pad_value=LABEL_PADDING_VALUE):
    """Pad the last dimension of a tensor to the target length."""
    current_length = tensor.shape[-1]
    if current_length >= target_length:
        return tensor[..., :target_length]
    padding = (0, target_length - current_length)
    return pad(tensor, padding, mode='constant', value=pad_value)