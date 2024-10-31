import utils
from data_utils import *
from torch.utils.data import DataLoader


if '__main__' == __name__:
	rank = 0
	n_gpus = 1
	
	hps = utils.get_hparams()
	train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
	train_sampler = DistributedBucketSampler(
		train_dataset,
		hps.train.batch_size,
		[32, 300, 400, 500, 600, 700, 800, 900, 1000],
		num_replicas=n_gpus,
		rank=rank,
		shuffle=True)
	collate_fn = TextAudioCollate()
	train_loader = DataLoader(dataset=train_dataset, num_workers=8, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler, shuffle=False)
	for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
		x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
		spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
		y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
	
	