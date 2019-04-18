from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import os
from io import open
from Model import util
import json
from Model.Encoder import Encoder
from Model.Attention import Attention

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

datafile = os.path.join("formatted.txt")
PAD_token, SOS_token, EOS_token = 0, 1, 2
MIN_COUNT = 3

voc, pairs = util.loadPrepareData(datafile)

with open('voc6.json', 'w') as fp:
    json.dump(voc.__dict__, fp)



pairs = util.trimRareWords(voc, pairs, MIN_COUNT)


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

#attn_model = 'dot'
attn_model = 'general'
hidden_size = 1200
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.1
batch_size = 128
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.00005
decoder_learning_ratio = 3.6
n_iteration = 300000
print_every = 10
save_every = 50000
loss_plt =[]


loadFilename = None
checkpoint_iter = 70000
loadFilename = '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size), '{}_checkpoint.tar'.format(checkpoint_iter)
checkpoint=None
if loadFilename:
    checkpoint = torch.load(loadFilename)
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
decoder = Attention.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.train()
decoder.train()

print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

print("Starting Training!")
util.trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, n_iteration, batch_size,
           print_every, save_every, clip, loadFilename, loss_plt, hidden_size, checkpoint)