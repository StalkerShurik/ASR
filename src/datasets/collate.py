import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    """
        dataset[i] : dict_keys(['audio', 'spectrogram', 'text', 'text_encoded', 'audio_path'])
    """

    batched_dataset_items = {}

    TEXT_ENCODED_MAX_SIZE = 0  # 2**8
    SPECTROGRAM_MAX_SIZE = 0  # 2**10
    AUDIO_MAX_SIZE = 0  # 2**18

    for i in range(len(dataset_items)):
        TEXT_ENCODED_MAX_SIZE = max(
            TEXT_ENCODED_MAX_SIZE, dataset_items[i]["text_encoded"].shape[1]
        )
        SPECTROGRAM_MAX_SIZE = max(
            SPECTROGRAM_MAX_SIZE, dataset_items[i]["spectrogram"].shape[2]
        )
        AUDIO_MAX_SIZE = max(AUDIO_MAX_SIZE, dataset_items[i]["audio"].shape[1])

    batched_dataset_items["spectrogram_length"] = []
    batched_dataset_items["text_encoded_length"] = []

    for i in range(len(dataset_items)):
        for key in dataset_items[i]:
            data_piece = dataset_items[i][key]

            if key == "text_encoded":
                batched_dataset_items["text_encoded_length"].append(
                    min(TEXT_ENCODED_MAX_SIZE, data_piece.shape[1])
                )
                data_piece = data_piece[:, :TEXT_ENCODED_MAX_SIZE]
                data_piece = F.pad(
                    data_piece,
                    pad=(0, TEXT_ENCODED_MAX_SIZE - data_piece.shape[1], 0, 0),
                )

            if key == "audio":
                data_piece = data_piece[:, :AUDIO_MAX_SIZE]
                data_piece = F.pad(
                    data_piece, pad=(0, AUDIO_MAX_SIZE - data_piece.shape[1], 0, 0)
                )

            if key == "spectrogram":
                batched_dataset_items["spectrogram_length"].append(
                    min(SPECTROGRAM_MAX_SIZE, data_piece.shape[2])
                )
                data_piece = data_piece[:, :, :SPECTROGRAM_MAX_SIZE]
                data_piece = F.pad(
                    data_piece,
                    pad=(0, SPECTROGRAM_MAX_SIZE - data_piece.shape[2], 0, 0, 0, 0),
                )

            if key not in batched_dataset_items.keys():
                if key in ["text_encoded", "audio", "spectrogram"]:
                    batched_dataset_items[key] = data_piece
                else:
                    batched_dataset_items[key] = []
                    batched_dataset_items[key].append(data_piece)
            else:
                if key in ["text_encoded", "audio", "spectrogram"]:
                    batched_dataset_items[key] = torch.cat(
                        (batched_dataset_items[key], data_piece)
                    )
                else:
                    batched_dataset_items[key].append(data_piece)

    batched_dataset_items["spectrogram_length"] = torch.tensor(
        batched_dataset_items["spectrogram_length"], dtype=torch.long
    )
    batched_dataset_items["text_encoded_length"] = torch.tensor(
        batched_dataset_items["text_encoded_length"], dtype=torch.long
    )

    return batched_dataset_items
