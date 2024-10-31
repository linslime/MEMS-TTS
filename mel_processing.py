import torch
import torch.utils.data


MAX_WAV_VALUE = 32768.0
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
	if torch.min(y) < -1.:
		print('min value is ', torch.min(y))
	if torch.max(y) > 1.:
		print('max value is ', torch.max(y))
	
	global hann_window
	dtype_device = str(y.dtype) + '_' + str(y.device)
	wnsize_dtype_device = str(win_size) + '_' + dtype_device
	if wnsize_dtype_device not in hann_window:
		hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)
	
	y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
	                            mode='reflect')
	y = y.squeeze(1)
	
	spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
	                  center=center, pad_mode='reflect', normalized=False, onesided=True)

	spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
	return spec
