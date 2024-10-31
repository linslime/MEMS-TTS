import torch


def intersperse(lst, item):
	result = [item] * (len(lst) * 2 + 1)
	result[1::2] = lst
	return result

def sequence_mask(length, max_length=None):
	if max_length is None:
		max_length = length.max()
	x = torch.arange(max_length, dtype=length.dtype, device=length.device)
	return x.unsqueeze(0) < length.unsqueeze(1)