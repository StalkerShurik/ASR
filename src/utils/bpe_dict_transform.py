import sentencepiece as spm

# Load a pre-trained SentencePiece model
sp = spm.SentencePieceProcessor(model_file="bpe.model")

vocab = [sp.id_to_piece(i) for i in range(sp.vocab_size())]

new_vocab = []

new_vocab.append("")
new_vocab.append(" ")

for word in vocab:
    if word in ["<unk>", "<sos/eos>", "<blk>"]:
        continue
    word = word.lower().replace("▁", " ")
    new_vocab.append(word)

f = open("bpe_vocab.vc", "w")

f.write(",".join(new_vocab))

exit()
