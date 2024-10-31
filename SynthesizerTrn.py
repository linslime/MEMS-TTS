from torch import nn
from transformer import Encoder, Decoder
import torch
import commons


class SynthesizerTrn(nn.Module):
	"""
	Synthesizer for Training
	"""
	
	def __init__(self, encoder_params, decoder_params):
		super().__init__()
		self.encoder = Encoder(*encoder_params)
		self.decoder = Decoder(*decoder_params)
	
	def forward(self, x, x_lengths, y, y_lengths, sid=None):
		x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
		output = self.encoder(x, x_mask)
		pass
	