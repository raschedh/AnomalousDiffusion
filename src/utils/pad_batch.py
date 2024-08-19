from torch.nn.utils.rnn import pad_sequence

LABEL_PADDING_VALUE = 99
FEATURE_PADDING_VALUE = 0

def pad_batch(batch):
    features, alpha_labels, k_labels, state_labels = zip(*batch)    
    features_padded = pad_sequence(features, batch_first=True, padding_value=FEATURE_PADDING_VALUE)
    alpha_labels_padded = pad_sequence(alpha_labels, batch_first=True, padding_value=LABEL_PADDING_VALUE)
    k_labels_padded = pad_sequence(k_labels, batch_first=True, padding_value=LABEL_PADDING_VALUE)
    state_labels_padded = pad_sequence(state_labels, batch_first=True, padding_value=LABEL_PADDING_VALUE)

    return features_padded, alpha_labels_padded, k_labels_padded, state_labels_padded