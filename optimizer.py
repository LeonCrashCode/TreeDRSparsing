import torch.optim as optim

def optimizer(args, parameters):
	if args.optimizer.lower() == "adam":
		return optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=args.learning_rate_f, weight_decay=args.weight_decay_f)
	elif args.optimizer.lower() == "sgd":
		return optim.SGD(filter(lambda p: p.requires_grad, parameters), lr=args.learning_rate_f, weight_decay=args.weight_decay_f)
	else:
		assert False, "no application for the optimizer"
