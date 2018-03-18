import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import utils


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # ------------
        # FILL THIS IN
        # ------------

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        # ------------
        # FILL THIS IN
        # ------------
        # z = ...
        # r = ...
        # g = ...
        # h_new = ...
        return h_new


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(GRUEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.opts = opts

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, inputs):
        """Forward pass of the encoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all time steps in the sequence. (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence. (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch. (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden

    def init_hidden(self, bs):
        """Creates a tensor of zeros to represent the initial hidden states
        of a batch of sequences.

        Arguments:
            bs: The batch size for the initial hidden state.

        Returns:
            hidden: An initial hidden state of all zeros. (batch_size x hidden_size)
        """
        return utils.to_var(torch.zeros(bs, self.hidden_size), self.opts.cuda)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size

        # ------------
        # FILL THIS IN
        # ------------

        # Create a two layer fully-connected network. Hint: Use nn.Sequential
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1

        # self.attention_network = ...

        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, annotations):
        """The forward pass of the attention mechanism.

        Arguments:
            hidden: The current decoder hidden state. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            output: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size, seq_len, hid_size = annotations.size()
        expanded_hidden = hidden.unsqueeze(1).expand_as(annotations)

        # ------------
        # FILL THIS IN
        # ------------

        # You are free to follow the code template below, or do it a different way,
        # as long as the output is correct.

        # concat = ...
        # reshaped_for_attention_net = ...
        # attention_net_output = ...
        # unnormalized_attention = ...  # Reshape attention net output to have dimension batch_size x seq_len x 1

        return self.softmax(unnormalized_attention)


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = MyGRUCell(input_size=hidden_size*2, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev, annotations):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            x: Input token indexes across a batch for a single time step. (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch. (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch. (batch_size x vocab_size)
            h_new: The new hidden states, across a batch. (batch_size x hidden_size)
            attention_weights: The weights applied to the encoder annotations, across a batch. (batch_size x encoder_seq_len x 1)
        """
        embed = self.embedding(x)    # batch_size x 1 x hidden_size
        embed = embed.squeeze(1)     # batch_size x hidden_size

        # ------------
        # FILL THIS IN
        # ------------
        # attention_weights = ...
        # context = ...
        # embed_and_context = ...
        # h_new = ...
        # output = ...
        return output, h_new, attention_weights


class NoAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(NoAttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev, inputs):
        """Forward pass of the non-attentional decoder RNN.

        Arguments:
            x: Input token indexes across a batch for a single time step. (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch. (batch_size x hidden_size)
            inputs: This is not used here. It just maintains consistency with the
                    interface used by the AttentionDecoder class.

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch. (batch_size x vocab_size)
            h_new: The new hidden states, across a batch. (batch_size x hidden_size)
            None: Used to maintain consistency with the interface of AttentionDecoder.
        """
        encoded = self.embedding(x)        # batch_size x 1 x hidden_size
        encoded = encoded.squeeze(1)       # batch_size x hidden_size
        h_new = self.rnn(encoded, h_prev)  # batch_size x hidden_size
        output = self.out(h_new)           # batch_size x vocab_size
        return output, h_new, None
