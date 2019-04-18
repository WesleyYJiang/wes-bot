from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import os
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from Model.Voc import Voc

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
MAX_LENGTH = 12
save_dir = os.path.join("data", "save")
PAD_token, SOS_token, EOS_token = 0, 1, 2


class GreedySearchDecoder(torch.jit.ScriptModule):
    def __init__(self, encoder, decoder, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._device = device
        self._SOS_token = SOS_token
        self._decoder_n_layers = decoder_n_layers

    __constants__ = ['_device', '_SOS_token', '_decoder_n_layers']

    @torch.jit.script_method
    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
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
voc = Voc("vocabulary")
checkpoint_iter = 175000
loadFilename = "175000_checkpoint.tar"

if loadFilename:
    #checkpoint = torch.load(loadFilename)
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
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
decoder = Decoder(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words)
test_seq_length = torch.LongTensor([test_seq.size()[0]])
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[:decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
traced_decoder = torch.jit.trace(decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs))
scripted_searcher = GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)

scripted_searcher.save("scripted_chatbot5.pth")