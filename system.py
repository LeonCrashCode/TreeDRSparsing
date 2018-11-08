import torch

def system_check_and_init(args):
	if args.gpu:
		assert torch.cuda.is_available(), "GPU is not available."
	if args.gpu:
		torch.cuda.manual_seed_all(args.random_seed)
	torch.manual_seed(args.random_seed)
