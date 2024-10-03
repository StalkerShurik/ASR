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

    text_encoded = dataset_items[0]["text_encoded"]
    audio = dataset_items[0]["audio"]
    spectrogram = dataset_items[0]["spectrogram"]

    print(f"COLLATE!!!!! {text_encoded.shape, audio.shape, spectrogram.shape}")

    """
        dataset[i] : dict_keys(['audio', 'spectrogram', 'text', 'text_encoded', 'audio_path'])
    """

    batched_dataset_items = {}

    TEXT_ENCODED_MAX_SIZE = 2**8
    SPECTROGRAM_MAX_SIZE = 2**10
    AUDIO_MAX_SIZE = 2**18

    for i in range(len(dataset_items)):
        for key in dataset_items[i]:
            data_piece = dataset_items[i][key]

            if key == "text_encoded":
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
                data_piece = data_piece[:, :, :SPECTROGRAM_MAX_SIZE]
                data_piece = F.pad(
                    data_piece,
                    pad=(0, SPECTROGRAM_MAX_SIZE - data_piece.shape[2], 0, 0, 0, 0),
                )

            # if key in ["text_encoded", "audio", "spectrogram"]:
            #     print(data_piece.shape)

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

    return batched_dataset_items
