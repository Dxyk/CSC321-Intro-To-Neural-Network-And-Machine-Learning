import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import utils

# CONSTANTS
d_float = torch.FloatTensor
d_long = torch.LongTensor


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # =============== Your Code Here ===============
        self.weight_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.weight_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.weight_g = nn.Linear(input_size + hidden_size, hidden_size)
        # ==============================================

    def forward(self, x, h_prev):
        """
        Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """
        # =============== Your Code Here ===============
        # r_t = \sigmoid(W_{ir}x_t + W_{hr}h_{t-1} + b_r)
        r = torch.sigmoid(self.weight_r(torch.cat(x, h_prev)))

        # z_t = \sigmoid(W_{iz}x_t + W_{hz}h_{t-1} + b_z)
        z = torch.sigmoid(self.weight_z(torch.cat(x, h_prev)))

        # g_t = tanh(W_{in}x_t + r_t * (W_{hn}h_{t-1} + b_g))
        g = torch.tanh(self.weight_g(torch.cat(x, r * h_prev)))

        # h_t = (1-z_t) * g + z * h_{t-1}
        h = (1 - z) * g + z * h_prev
        return h
        # ==============================================

        # =============== Your Code Here ===============
        r = torch.sigmoid(self.weight_r(torch.cat((x, h_prev), 1)))
        z = torch.sigmoid(self.weight_z(torch.cat((x, h_prev), 1)))
        g = F.tanh(self.weight_g(torch.cat((x, r * h_prev), 1)))
        h_new = (1 - z) * g + z * h_prev
        return h_new
        # ==============================================


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
            inputs: Input token indexes across a batch for all time steps in the sequence.
                (batch_size x seq_len)

        Returns:
            annotations: The hidden states computed at each step of the input sequence.
                (batch_size x seq_len x hidden_size)
            hidden: The final hidden state of the encoder, for each sequence in a batch.
                (batch_size x hidden_size)
        """

        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)  # batch_size x seq_len x hidden_size
        annotations = []

        for i in range(seq_len):
            x = encoded[:, i, :]  # Get the current time step, across the whole batch
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

        # =============== Your Code Here ===============
        # function f, parameterized as a two-layer fully-connected network with a ReLU activation.
        # \tilda{a}_i = f(h^{dec}_{t-1}, h^{enc}_t)
        #             = W_2(\max(0, W_1[h^{dec}_{t-1};h^{enc}_t]) + b_1)) + b_2
        # where [h^{dec}_{t-1};h^{enc}_t] = np.cat(h^{dec}_{t-1}, h^{enc}_t)
        #
        # Input: batch_size * sequence_size (num_hidden) * hidden_size
        # Output: batch_size * sequence_size * 1
        self.attention_network = nn.Sequential(
            # W_1[h^{dec}_{t-1};h^{enc}_t]) + b_1
            nn.Linear(hidden_size * 2, hidden_size),

            # max(0, W_1[h^{dec}_{t-1};h^{enc}_t]) + b_1)
            nn.ReLU(),

            # W_2(\max(0, W_1[h^{dec}_{t-1};h^{enc}_t]) + b_1)) + b_2
            nn.Linear(hidden_size, 1)
        )

        self.softmax = nn.Softmax(dim=1)
        # ==============================================

    def forward(self, hidden, annotations):
        """
        The forward pass of the attention mechanism.

        Arguments:
            hidden: The current decoder hidden state.
                (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input sequence.
                (batch_size x seq_len x hidden_size)

        Returns:
            output: Normalized attention weights for each encoder hidden state.
                (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size, seq_len, hid_size = annotations.size()

        # =============== Your Code Here ===============
        # unsqueeze(dim): add 1 dim to the dim given
        #   e.g. >>> Tensor([1, 2, 3, 4]).unsqueeze(1)  # (4) -> (1 * 4)
        #        Tensor([[1], [2], [3], [4]])
        # expand_as(tensor): equivalent to expand(tensor.size())
        # expand(*size): expand the tensor to the provided size in the corresponding dimensions
        #   e.g. >>> Tensor([[1], [2], [3]]).expand(3, 4)
        #        1  1  1  1
        #        2  2  2  2
        #        3  3  3  3
        #        [torch.FloatTensor of size 3x4]
        expanded_hidden = hidden.unsqueeze(1).expand_as(annotations)

        # cat(seq, dim): concatenate the given sequence in the given dimension
        concat_hidden = torch.cat((expanded_hidden, annotations), 2)

        # tensor.view(size): view the tensor as the given size
        #   -1: default according to the other dim size
        reshaped_for_attention_net = concat_hidden.view((-1, 2 * hid_size))

        attention_net_output = self.attention_network(reshaped_for_attention_net)

        # Reshape attention net output to have dimension batch_size x seq_len x 1
        unnormalized_attention = attention_net_output.view(batch_size, seq_len, 1)

        return self.softmax(unnormalized_attention)
        # ==============================================


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = MyGRUCell(input_size=hidden_size * 2, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_prev, annotations):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            x: Input token indexes across a batch for a single time step.
               (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch.
                (batch_size x hidden_size)
            annotations: The encoder hidden states for each step of the input sequence.
                (batch_size x seq_len x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch.
                (batch_size x vocab_size)
            h_new: The new hidden states, across a batch. (batch_size x hidden_size)
            attention_weights: The weights applied to the encoder annotations, across a batch.
                (batch_size x encoder_seq_len x 1)
        """
        embed = self.embedding(x)  # batch_size x 1 x hidden_size
        embed = embed.squeeze(1)  # batch_size x hidden_size

        # =============== Your Code Here ===============
        # TODO
        attention_weights = self.attention(embed, annotations)

        attention_weights_t = torch.transpose(attention_weights, 1, 2)  # (B, S->1, 1->S)
        context = torch.bmm(attention_weights_t, annotations)  # (B, 1, S), (B, S, H)
        # context = torch.mm(torch.t(torch.squeeze(attention_weights, 2)), embed)
        # print "context:", context.size()  # (B, 1, H)
        # print x.size()
        context = torch.squeeze(context, 1)
        # print "context:", context.size()  # (B, H)
        embed_and_context = torch.cat((context.type(d_float), embed.type(d_float)), 1)
        h_new = self.rnn(embed_and_context.type(d_float), h_prev.type(d_float))
        output = self.out(h_new)
        return output, h_new, attention_weights
        # ==============================================


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
            x: Input token indexes across a batch for a single time step.
                (batch_size x 1)
            h_prev: The hidden states from the previous step, across a batch.
                (batch_size x hidden_size)
            inputs: This is not used here. It just maintains consistency with the
                    interface used by the AttentionDecoder class.

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch.
                (batch_size x vocab_size)
            h_new: The new hidden states, across a batch.
                (batch_size x hidden_size)
            None: Used to maintain consistency with the interface of AttentionDecoder.
        """
        encoded = self.embedding(x)  # batch_size x 1 x hidden_size
        encoded = encoded.squeeze(1)  # batch_size x hidden_size
        h_new = self.rnn(encoded, h_prev)  # batch_size x hidden_size
        output = self.out(h_new)  # batch_size x vocab_size
        return output, h_new, None
