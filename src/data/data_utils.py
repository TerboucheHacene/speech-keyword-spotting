import torch


labels = [
    "marvin",
    "off",
    "left",
    "one",
    "cat",
    "on",
    "bed",
    "house",
    "sheila",
    "down",
    "happy",
    "visual",
    "five",
    "stop",
    "dog",
    "wow",
    "seven",
    "zero",
    "backward",
    "no",
    "eight",
    "three",
    "four",
    "tree",
    "nine",
    "go",
    "bird",
    "right",
    "yes",
    "up",
    "follow",
    "learn",
    "two",
    "forward",
    "six",
]


def label_to_index(word, labels=labels):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index, labels=labels):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # # # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]
        # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


def collate_fn_spec(batch):
    # A data tuple has the form:
    # # # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    # Gather in lists, and encode labels as indices
    for spec, label in batch:
        ncrops, c, h, w = spec.size()
        tensors += [spec]
        targets += [label.repeat(ncrops)]
        # Group the list of tensors into a batched tensor
    tensors = torch.cat(tensors, dim=0)
    targets = torch.cat(targets, dim=0)
    return tensors, targets
