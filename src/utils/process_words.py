f_out = open(
    "/Users/alexandergaponov/.cache/torch/hub/torchaudio/decoder-assets/librispeech/lexicon_processed.txt",
    "w",
)

with open(
    "/Users/alexandergaponov/.cache/torch/hub/torchaudio/decoder-assets/librispeech/lexicon.txt"
) as f:
    for line in f:
        f_out.write(line.replace(" '", "").replace("'", "").replace("|", " "))
