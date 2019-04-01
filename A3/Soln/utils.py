"""Utility functions.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

mpl.use('Agg')


def string_to_index_list(s, char_to_index, end_token):
    """
    Converts a sentence into a list of indexes (for each character).
    """
    index_lst = [char_to_index[char] for char in s]
    return index_lst + [end_token]  # Adds the end token to each index list


def translate_sentence(sentence, encoder, decoder, idx_dict, opts):
    """
    Translates a sentence from English to Pig-Latin, by splitting the sentence into
    words (whitespace-separated), running the encoder-decoder model to translate each
    word independently, and then stitching the words back together with spaces between them.
    """
    return ' '.join(
        [translate(word, encoder, decoder, idx_dict, opts) for word in sentence.split()])


def translate(input_string, encoder, decoder, idx_dict, opts):
    """
    Translates a given string from English to Pig-Latin.
    """
    char_to_index = idx_dict['char_to_index']
    index_to_char = idx_dict['index_to_char']
    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']

    # print "char_to_index:\n", char_to_index
    # print "index_to_char:\n", index_to_char
    # print "start_token:\n", start_token
    # print "end_token:\n", end_token

    max_generated_chars = 20
    gen_string = ''

    indexes = string_to_index_list(input_string, char_to_index, end_token)
    indexes = to_var(torch.LongTensor(indexes).unsqueeze(0),
                     opts.cuda)  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_last_hidden = encoder(indexes)

    decoder_hidden = encoder_last_hidden
    decoder_input = to_var(torch.LongTensor([[start_token]]), opts.cuda)  # For BS = 1

    for i in range(max_generated_chars):
        decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden,
                                                                    encoder_annotations)
        ni = F.softmax(decoder_output, dim=1).data.max(1)[1]  # LongTensor of size 1
        ni = ni[0].item()

        if ni == end_token:
            break
        else:
            gen_string += index_to_char[ni]
            decoder_input = to_var(torch.LongTensor([[ni]]), opts.cuda)

    return gen_string


def visualize_attention(input_string, encoder, decoder, idx_dict, opts, save='save.pdf'):
    """Generates a heatmap to show where attention is focused in each decoder step.
    """

    char_to_index = idx_dict['char_to_index']
    index_to_char = idx_dict['index_to_char']
    start_token = idx_dict['start_token']
    end_token = idx_dict['end_token']

    max_generated_chars = 20
    gen_string = ''

    all_attention_weights = []

    indexes = string_to_index_list(input_string, char_to_index, end_token)
    indexes = to_var(torch.LongTensor(indexes).unsqueeze(0),
                     opts.cuda)  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_hidden = encoder(indexes)

    decoder_hidden = encoder_hidden
    decoder_input = to_var(torch.LongTensor([[start_token]]), opts.cuda)  # For BS = 1

    produced_end_token = False

    for i in range(max_generated_chars):
        decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden,
                                                                    encoder_annotations)

        ni = F.softmax(decoder_output, dim=1).data.max(1)[1]  # LongTensor of size 1
        ni = ni.item()

        all_attention_weights.append(attention_weights.squeeze().data.cpu().numpy())

        if ni == end_token:
            produced_end_token = True
            break
        else:
            gen_string += index_to_char[ni]
            decoder_input = to_var(torch.LongTensor([[ni]]), opts.cuda)

    attention_weights_matrix = np.stack(all_attention_weights)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_weights_matrix.T, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + list(input_string) + ['EOS'], rotation=90)
    ax.set_xticklabels([''] + list(gen_string) + (['EOS'] if produced_end_token else []))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(save)

    plt.close(fig)

    return gen_string


def to_var(tensor, cuda):
    """
    Wraps a Tensor in a Variable, optionally placing it on the GPU.

        Arguments:
            tensor: A Tensor object.
            cuda: A boolean flag indicating whether to use the GPU.

        Returns:
            A Variable object, on the GPU if cuda==True.
    """
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
