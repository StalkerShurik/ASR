import torch
from torch import nn
from torch.nn import Sequential


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride, pad, activation: str = "ReLU"):
        super().__init__()

        self.model = Sequential(
            nn.Conv2d(cin, cout, kernel, stride, pad),
            nn.BatchNorm2d(cout),
            getattr(nn, activation)(inplace=True),
        )

    def forward(self, input):
        return self.model(input)


class ConvolutionalExtractor(nn.Module):
    """
    one convolutional block
    """

    def __init__(
        self, n_blocks, n_channels=32, wider_n_channels=96
    ):  # n_blocks in [1,3] according to the article
        super().__init__()

        if n_blocks not in [2, 3]:
            raise Exception("Number of convolutional blocks must be 2 or 3")

        self.n_blocks = n_blocks
        self.channels = n_channels
        self.wider_channels = wider_n_channels

        self.head_block = ConvBlock(1, n_channels, (41, 11), (2, 2), 0)

        self.body_block = ConvBlock(n_channels, n_channels, (21, 11), (2, 1), 0)

        if n_blocks == 3:
            self.tail_block = ConvBlock(
                n_channels, wider_n_channels, (21, 11), (2, 1), 0
            )

    def forward(self, input):
        output = self.body_block(self.head_block(input))

        if self.n_blocks == 3:
            output = self.tail_block(output)

        return output


class RNNBlock(nn.Module):
    def __init__(
        self, input_size, rnn, hidden_size, num_layers, dropout, need_batchnorm=False
    ):
        super().__init__()
        self.input_size = input_size
        self.rnn = rnn
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.bidirectional = True
        self.need_batchnorm = need_batchnorm

        self.rnn = getattr(nn, rnn)(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
            bias=False,
        )

        if self.need_batchnorm:
            self.batchnorm = nn.BatchNorm1d(hidden_size)

    def forward(self, input):
        output = self.rnn(input)[0]
        bs, tm, ft = output.shape
        output = output.reshape((bs, tm, 2, -1))
        output = output.sum(-2)

        if self.need_batchnorm:
            output = self.batchnorm(output.permute((0, 2, 1))).permute((0, 2, 1))
        return output


class RNNProcessor(nn.Module):
    def __init__(
        self, n_blocks, input_size, rnn, hidden_size, num_layers, dropout
    ) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.model = nn.Sequential(
            *[
                RNNBlock(
                    input_size if i == 0 else hidden_size,
                    rnn,
                    hidden_size,
                    num_layers,
                    dropout,
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, input):
        return self.model(input)


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_conv_blocks,
        n_rnn_blocks,
        rnn,
        rnn_hidden_size,
        num_rnn_layers,
        rnn_dropout=0,
        conv_n_channels=32,
        conv_tail_channels_growth=96,
        *args,
        **kwargs
    ):
        super().__init__()

        self.convolution_extractor = ConvolutionalExtractor(
            n_conv_blocks, conv_n_channels, conv_tail_channels_growth
        )

        if n_conv_blocks == 2:
            self.conv_out_features_size = 384
        else:
            raise Exception("look deepspeech2 conv_out_features_size")

        self.rnn_processor = RNNProcessor(
            n_rnn_blocks,
            self.conv_out_features_size,
            rnn,
            rnn_hidden_size,
            num_rnn_layers,
            rnn_dropout,
        )

        self.head = nn.Linear(rnn_hidden_size, n_tokens, bias=False)

    def forward(self, spectrogram, spectrogram_length, **bacth):
        spectrogram = spectrogram.unsqueeze(1)

        output = self.convolution_extractor(spectrogram)

        bs, ch, ft, seq_len = output.shape

        output = self.rnn_processor(
            output.reshape(bs, ch * ft, seq_len).permute((0, 2, 1))
        )

        output = self.head(output)

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def lengths_after_conv(
        self, input_lengths, kernel_size, padding, stride, dillation
    ):
        tmp = torch.tensor(
            input_lengths + 2 * padding - dillation * (kernel_size - 1) - 1
        )
        return torch.floor(tmp / stride + 1)

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """

        convs = []
        convs.append(self.convolution_extractor.head_block.model[0])
        convs.append(self.convolution_extractor.body_block.model[0])

        if self.convolution_extractor.n_blocks == 3:
            convs.append(self.convolution_extractor.tail_block.model[0])

        for conv in convs:
            input_lengths = self.lengths_after_conv(
                input_lengths,
                conv.kernel_size[1],
                conv.padding[1],
                conv.stride[1],
                conv.dilation[1],
            ).type(torch.int32)
        return input_lengths
