import argparse
from system import system_check_and_init
from utils import read_input
from utils import get_singleton_dict
from utils import input2instance
from utils import read_tree
from utils import tree2action as tree2action
from utils import get_same_lemma

from dictionary.vocabulary import vocabulary
from dictionary.PretrainedEmb import PretrainedEmb
from representation.sentence_rep import sentence_rep

from encoder.bilstm import comb_encoder as enc 
from decoder.lstm import decoder as dec

from utils import get_k_scope
from utils import get_p_max
from constraints.constraints import struct_constraints as cstn_step1
from constraints.constraints import relation_constraints as cstn_step2
from constraints.constraints import variable_constraints as cstn_step3

import torch
from optimizer import optimizer

import types
import sys

def run_train(args):
	system_check_and_init(args)
	if args.gpu:
		print "GPU available"
	else:
		print "CPU only"

	word_v = vocabulary()
	char_v = vocabulary()
	actn_v = vocabulary(UNK=False)
	pretrain = PretrainedEmb(args.pretrain_path)

	actn_v.toidx("<START>")
	actn_v.toidx("<END>")
	for i in range(args.X_l):
		actn_v.toidx("X"+str(i+1))
	for i in range(args.E_l):
		actn_v.toidx("E"+str(i+1))
	for i in range(args.S_l):
		actn_v.toidx("S"+str(i+1))
	for i in range(args.T_l):
		actn_v.toidx("T"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1))
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1)+"(")
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1)+"(")
	actn_v.toidx("CARD_NUMBER")
	actn_v.toidx("TIME_NUMBER")
	actn_v.toidx(")")

	#print actn_v.size()
	actn_v.read_file(args.action_dict_path)
	#print actn_v.size()
	actn_v.freeze()
	#instances
	train_input = read_input(args.train_input)

	#print train_input[0]
	train_comb = [ get_same_lemma(x[1]) for x in train_input]
	dev_input = read_input(args.dev_input)
	dev_comb = [ get_same_lemma(x[1]) for x in dev_input]

	singleton_idx_dict, word_dict, word_v = get_singleton_dict(train_input, word_v)
	extra_vl = [ vocabulary() for i in range(len(train_input[0])-1)]	
	train_instance, word_v, char_v, extra_vl = input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, "train")
	word_v.freeze()
	char_v.freeze()
	for i in range(len(extra_vl)):
		extra_vl[i].freeze()
	dev_instance, word_v, char_v, extra_vl = input2instance(dev_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")

	train_output = read_tree(args.train_action)
	#print train_output[0]
	#dev_output = read_output(args.dev_action)
	train_action = tree2action(train_output, actn_v)
	#dev_actoin, actn_v = output2action(dev_output, actn_v)
	#print train_action[0][0]
	#print train_action[0][1]
	#print train_action[0][2]
	print "word vocabulary size:", word_v.size()
	word_v.dump(args.model_path_base+"/word.list")
	print "char vocabulary size:", char_v.size()
	if args.use_char:
		char_v.dump(args.model_path_base+"/char.list")
	print "pretrain vocabulary size:", pretrain.size()
	extra_vl_size = []
	for i in range(len(extra_vl)):
		print "extra", i, "vocabulary size:", extra_vl[i].size()
		extra_vl[i].dump(args.model_path_base+"/extra."+str(i+1)+".list")
		extra_vl_size.append(extra_vl[i].size())
	print "action vocaluary size:", actn_v.size()
	
	#actn_v.dump()

	# neural components
	input_representation = sentence_rep(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
	encoder = None
	#if args.encoder == "BILSTM":
	#	from encoder.bilstm import encoder as enc
	#elif args.encoder == "Transformer":
	#	from encoder.transformer import encoder as enc
	encoder = enc(args)
	assert encoder, "please specify encoder type"
	
	#check dict to get index
	#BOX DISCOURSE RELATION PREDICATE CONSTANT
	starts = []
	ends = []
	lines = []
	for line in open(args.action_dict_path):
		line = line.strip()
		if line == "###":
			starts.append(actn_v.toidx(lines[0]))
			ends.append(actn_v.toidx(lines[-1]))
			lines = []
		else:
			if line[0] == "#":
				continue
			lines.append(line)

	#mask = Mask(args, actn_v, starts, ends)
	decoder = dec(actn_v.size(), args, actn_v)

	if args.gpu:
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		input_representation = input_representation.cuda()
	
	#training process
	model_parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(input_representation.parameters())
	
	model_optimizer = optimizer(args, model_parameters)
	lr = args.learning_rate_f

	i = len(train_instance)
	check_iter = 0
	check_loss1 = 0
	check_loss2 = 0
	check_loss3 = 0
	bscore = -1
	epoch = -1
	while True:
		"""
		for p in model_parameters:
			if p.grad is not None:
				p.grad.detach_()
				p.grad.zero_()
		"""
		model_optimizer.zero_grad()
		if i == len(train_instance):
			i = 0
			epoch += 1
			lr = args.learning_rate_f / (1 + epoch * args.learning_rate_decay_f)

		check_iter += 1
		input_t = input_representation(train_instance[i], singleton_idx_dict=singleton_idx_dict, train=True)
		enc_rep_t, hidden_t = encoder(input_t, train_comb[i], train=True)
		#step 1
		hidden_step1 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
		loss_t1, hidden_rep_t, hidden_step1 = decoder(train_action[i][0], hidden_step1, enc_rep_t, train=True, constraints=None, opt=1)
		check_loss1 += loss_t1.data.tolist()
		
		#step 2
		idx = 0
		hidden_step2 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
		train_action_step2 = []
		for j in range(len(train_action[i][0])): #<START> DRS( P1(
			tok = train_action[i][0][j]
			if actn_v.totok(tok) in ["DRS(", "SDRS("]:
				train_action_step2.append([hidden_rep_t[j], train_action[i][1][idx]])
				idx += 1
		assert idx == len(train_action[i][1])
		loss_t2, hidden_rep_t, hidden_step2 = decoder(train_action_step2, hidden_step2, enc_rep_t, train=True, constraints=None, opt=2)
		check_loss2 += loss_t2.data.tolist()
		
		#step 3
		flat_train_action = [0] # <START>
		for l in train_action[i][1]:
			flat_train_action += l
		#print flat_train_action
		idx = 0
		hidden_step3 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
		train_action_step3 = []
		for j in range(len(flat_train_action)):
			tok = flat_train_action[j]
			#print tok
			if (type(tok) == types.StringType and tok[-1] == "(") or actn_v.totok(tok)[-1] == "(":
				train_action_step3.append([hidden_rep_t[j], train_action[i][2][idx]])
				idx += 1
		#print i, idx, len(train_action[i][2])
		assert idx == len(train_action[i][2])
		loss_t3, hidden_rep_t, hidden_step3 = decoder(train_action_step3, hidden_step3, enc_rep_t, train=True, constraints=None, opt=3)
		check_loss3 += loss_t3.data.tolist()

		if check_iter % args.check_per_update == 0:
			print('epoch %.6f : structure %.10f, relation %.10f, variable %.10f, [lr: %.6f]' % (check_iter*1.0/len(train_instance), check_loss1*1.0 / args.check_per_update, check_loss2*1.0 / args.check_per_update, check_loss3*1.0 / args.check_per_update, lr))
			check_loss1 = 0
			check_loss2 = 0
			check_loss3 = 0
		
		i += 1
		loss_t = loss_t1 + loss_t2 + loss_t3
		loss_t.backward()
		torch.nn.utils.clip_grad_value_(model_parameters, 5)

		#model_optimizer.step()
		"""
		for p in model_parameters:
			if p.requires_grad:
				p.data.add_(-lr, p.grad.data)
		"""
		model_optimizer.step()
		
		if check_iter % args.eval_per_update == 0:
			torch.save({"encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "input_representation": input_representation.state_dict()}, args.model_path_base+"/model"+str(int(check_iter/args.eval_per_update)))

			cstns1 = cstn_step1(actn_v, args)
			cstns2 = cstn_step2(actn_v, args, starts, ends)
			cstns3 = cstn_step3(actn_v, args)
			with open(args.dev_output_path_base+"/"+str(int(check_iter/args.eval_per_update)), "w") as w:
				for j, instance in enumerate(dev_instance):
					print j
					dev_input_t = input_representation(instance, singleton_idx_dict=None, train=False)
					dev_enc_rep_t, dev_hidden_t= encoder(dev_input_t, dev_comb[j], train=False)

					#step 1
					dev_hidden_step1 = (dev_hidden_t[0].view(args.action_n_layer, 1, -1), dev_hidden_t[1].view(args.action_n_layer, 1, -1))
					cstns1.reset()
					dev_output_step1, dev_hidden_rep_step1, dev_hidden_step1 = decoder(actn_v.toidx("<START>"), dev_hidden_step1, dev_enc_rep_t, train=False, constraints=cstns1, opt=1)
					#print dev_output_step1

					#step 2
					dev_output_step2 = []
					dev_hidden_rep_step2 = []
					dev_hidden_step2 = (dev_hidden_t[0].view(args.action_n_layer, 1, -1), dev_hidden_t[1].view(args.action_n_layer, 1, -1))
					cstns2.reset_length(len(instance[0])-2) # <s> </s>
					for k in range(len(dev_output_step1)): # DRS( P1(
						act1 = dev_output_step1[k]
						if actn_v.totok(act1) in ["DRS(", "SDRS("]:
							cstns2.reset_condition(act1)
							one_dev_output_step2, one_dev_hidden_rep_step2, dev_hidden_step2 = decoder(dev_hidden_rep_step1[k+1], dev_hidden_step2, dev_enc_rep_t, train=False, constraints=cstns2, opt=2)
							dev_output_step2.append(one_dev_output_step2)
							dev_hidden_rep_step2.append(one_dev_hidden_rep_step2)
					#print dev_output_step2

					#step 3
					k_scope = get_k_scope(dev_output_step1, actn_v)
					p_max = get_p_max(dev_output_step1, actn_v)
					dev_output_step3 = []
					dev_hidden_step3 = (dev_hidden_t[0].view(args.action_n_layer, 1, -1), dev_hidden_t[1].view(args.action_n_layer, 1, -1))
					cstns3.reset(p_max)
					k = 0
					sdrs_idx = 0
					for act1 in dev_output_step1:
						if actn_v.totok(act1) in ["DRS(", "SDRS("]:
							if actn_v.totok(act1) == "SDRS(":
								cstns3.reset_condition(act1, k_scope[sdrs_idx])
								sdrs_idx += 1
							else:
								cstns3.reset_condition(act1)
							for kk in range(len(dev_output_step2[k])-1): # rel( rel( )
								act2 = dev_output_step2[k][kk]
								cstns3.reset_relation(act2)
								one_dev_output_step3, _, dev_hidden_step3 = decoder(dev_hidden_rep_step2[k][kk+1], dev_hidden_step3, dev_enc_rep_t, train=False, constraints=cstns3, opt=3)
								dev_output_step3.append(one_dev_output_step3)
							k += 1
					#print dev_output_step3

					# write file
					dev_output = []
					k = 0
					kk = 0
					for act1 in dev_output_step1:
						dev_output.append(actn_v.totok(act1))
						if dev_output[-1] in ["DRS(", "SDRS("]:
							for act2 in dev_output_step2[k][:-1]:
								if act2 >= actn_v.size():
									dev_output.append("$"+str(act2-actn_v.size())+"(")
								else:
									dev_output.append(actn_v.totok(act2))
								for act3 in dev_output_step3[kk]:
									dev_output.append(actn_v.totok(act3))
								kk += 1
							k += 1
					w.write(" ".join(dev_output) + "\n")
					w.flush()
				w.close()
			#exit(1)

def run_test(args):
	word_v = vocabulary()
	word_v.read_file(args.model_path_base+"/word.list")
	word_v.freeze()

	char_v = vocabulary()
	if args.use_char:
		char_v.read_file(args.model_path_base+"/char.list")
		char_v.freeze()

	actn_v = vocabulary(UNK=False)
	pretrain = PretrainedEmb(args.pretrain_path)

	actn_v.toidx("<START>")
	actn_v.toidx("<END>")
	for i in range(args.X_l):
		actn_v.toidx("X"+str(i+1))
	for i in range(args.E_l):
		actn_v.toidx("E"+str(i+1))
	for i in range(args.S_l):
		actn_v.toidx("S"+str(i+1))
	for i in range(args.T_l):
		actn_v.toidx("T"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1))
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1)+"(")
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1)+"(")
	actn_v.toidx("CARD_NUMBER")
	actn_v.toidx("TIME_NUMBER")
	actn_v.toidx(")")

	actn_v.read_file(args.action_dict_path)
	actn_v.freeze()

	test_input = read_input(args.test_input)
	test_comb = [ get_same_lemma(x[1]) for x in test_input]

	extra_vl = [ vocabulary() for i in range(len(test_input[0])-1)]
	for i in range(len(test_input[0])-1):
		extra_vl[i].read_file(args.model_path_base+"/extra."+str(i+1)+".list")
		extra_vl[i].freeze()

	print "word vocabulary size:", word_v.size()
	print "char vocabulary size:", char_v.size() 
	print "pretrain vocabulary size:", pretrain.size()
	extra_vl_size = []
	for i in range(len(extra_vl)):
		print "extra", i, "vocabulary size:", extra_vl[i].size()
		extra_vl_size.append(extra_vl[i].size())
	print "action vocaluary size:", actn_v.size() 

	input_representation = sentence_rep(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
	encoder = None
	#if args.encoder == "BILSTM":
	#	from encoder.bilstm import encoder as enc
	#elif args.encoder == "Transformer":
	#	from encoder.transformer import encoder as enc
	encoder = enc(args)
	assert encoder, "please specify encoder type"
	
	#check dict to get index
	#BOX DISCOURSE RELATION PREDICATE CONSTANT
	starts = []
	ends = []
	lines = []
	for line in open(args.action_dict_path):
		line = line.strip()
		if line == "###":
			starts.append(actn_v.toidx(lines[0]))
			ends.append(actn_v.toidx(lines[-1]))
			lines = []
		else:
			if line[0] == "#":
				continue
			lines.append(line)

	#mask = Mask(args, actn_v, starts, ends)
	decoder = dec(actn_v.size(), args, actn_v)

	check_point = torch.load(args.model_path_base+"/model")
	encoder.load_state_dict(check_point["encoder"])
	decoder.load_state_dict(check_point["decoder"])
	input_representation.load_state_dict(check_point["input_representation"])

	if args.gpu:
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		input_representation = input_representation.cuda()
	

	
	test_instance, word_v, char_v, extra_vl = input2instance(test_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")

	cstns1 = cstn_step1(actn_v, args)
	cstns2 = cstn_step2(actn_v, args, starts, ends)
	cstns3 = cstn_step3(actn_v, args)
	with open(args.test_output, "w") as w:
		for j, instance in enumerate(test_instance):
			print j
			test_input_t = input_representation(instance, singleton_idx_dict=None, train=False)
			test_enc_rep_t, test_hidden_t= encoder(test_input_t, test_comb[j], train=False)

			#step 1
			test_hidden_step1 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			cstns1.reset()
			test_output_step1, test_hidden_rep_step1, test_hidden_step1 = decoder(actn_v.toidx("<START>"), test_hidden_step1, test_enc_rep_t, train=False, constraints=cstns1, opt=1)
			#print [actn_v.totok(x) for x in test_output_step1]
			#print test_hidden_rep_step1[1]
			#exit(1)
			#step 2
			test_output_step2 = []
			test_hidden_rep_step2 = []
			test_hidden_step2 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			cstns2.reset_length(len(instance[0])-2) # <s> </s>
			for k in range(len(test_output_step1)): # DRS( P1(
				act1 = test_output_step1[k]
				if actn_v.totok(act1) in ["DRS(", "SDRS("]:
					cstns2.reset_condition(act1)
					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2 = decoder(test_hidden_rep_step1[k+1], test_hidden_step2, test_enc_rep_t, train=False, constraints=cstns2, opt=2)
					test_output_step2.append(one_test_output_step2)
					test_hidden_rep_step2.append(one_test_hidden_rep_step2)
					#print test_hidden_step2
					#print one_test_hidden_rep_step2
					#print test_hidden_step2
					#exit(1)
			#print test_output_step2
			#step 3
			k_scope = get_k_scope(test_output_step1, actn_v)
			p_max = get_p_max(test_output_step1, actn_v)
			
			test_output_step3 = []
			test_hidden_step3 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			cstns3.reset(p_max)
			k = 0
			sdrs_idx = 0
			for act1 in test_output_step1:
				if actn_v.totok(act1) in ["DRS(", "SDRS("]:
					if actn_v.totok(act1) == "SDRS(":
						cstns3.reset_condition(act1, k_scope[sdrs_idx])
						sdrs_idx += 1
					else:
						cstns3.reset_condition(act1)
					for kk in range(len(test_output_step2[k])-1): # rel( rel( )
						act2 = test_output_step2[k][kk]
						#if act2 >= actn_v.size():
						#	print "$"+str(act2 - actn_v.size())+"("
						#else:
						#	print actn_v.totok(act2)
							
						cstns3.reset_relation(act2)
						#print test_hidden_rep_step2[k][kk+1]
						#print test_hidden_step3
						#print "==============================="
						one_test_output_step3, _, test_hidden_step3 = decoder(test_hidden_rep_step2[k][kk+1], test_hidden_step3, test_enc_rep_t, train=False, constraints=cstns3, opt=3)
						test_output_step3.append(one_test_output_step3)
						#exit(1)
					k += 1
			#print test_output_step3

			# write file
			test_output = []
			k = 0
			kk = 0
			for act1 in test_output_step1:
				test_output.append(actn_v.totok(act1))
				if test_output[-1] in ["DRS(", "SDRS("]:
					for act2 in test_output_step2[k][:-1]:
						if act2 >= actn_v.size():
							test_output.append("$"+str(act2-actn_v.size())+"(")
						else:
							test_output.append(actn_v.totok(act2))
						for act3 in test_output_step3[kk]:
							test_output.append(actn_v.totok(act3))
						kk += 1
					k += 1
			w.write(" ".join(test_output) + "\n")
			w.flush()
		w.close()

def run_check(args):

	import re
	actn_v = vocabulary(UNK=False)

	actn_v.toidx("<START>")
	actn_v.toidx("<END>")
	for i in range(args.X_l):
		actn_v.toidx("X"+str(i+1))
	for i in range(args.E_l):
		actn_v.toidx("E"+str(i+1))
	for i in range(args.S_l):
		actn_v.toidx("S"+str(i+1))
	for i in range(args.T_l):
		actn_v.toidx("T"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1))
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1)+"(")
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1)+"(")
	actn_v.toidx("CARD_NUMBER")
	actn_v.toidx("TIME_NUMBER")
	actn_v.toidx(")")

	print actn_v.size()
	actn_v.read_file(args.action_dict_path)
	print actn_v.size()
	actn_v.freeze()

	train_input = read_input(args.train_input)
	train_output = read_tree(args.train_action)
	#print train_output[0]
	#dev_output = read_output(args.dev_action)
	train_action = tree2action(train_output, actn_v)
	#dev_actoin, actn_v = output2action(dev_output, actn_v)
	print "action vocaluary size:", actn_v.size()

	#check dict to get index
	#BOX DISCOURSE RELATION PREDICATE CONSTANT
	starts = []
	ends = []
	lines = []
	for line in open(args.action_dict_path):
		line = line.strip()
		if line == "###":
			starts.append(actn_v.toidx(lines[0]))
			ends.append(actn_v.toidx(lines[-1]))
			lines = []
		else:
			if line[0] == "#":
				continue
			lines.append(line)

	cstns1 = cstn_step1(actn_v, args)
	cstns2 = cstn_step2(actn_v, args, starts, ends)
	cstns3 = cstn_step3(actn_v, args)

	line = 0
	for i in range(len(train_action)):
		action_step1 = train_action[i][0]
		action_step2 = train_action[i][1]
		action_step3 = train_action[i][2]
		#print [[actn_v.totok(x) for x in action] for action in action_step3]
		cstns2.reset_length(len(train_input[i][0])-2)

		#print action_step2
		line += 1
		print line
		cstns1.reset()
		#processed_act = []
		idx = 0
		idx2 = 0

		# k_scope
		stack = []
		k_scope = {}
		sdrs_idx = 0
		for act in action_step1[1:]:
			#print stack
			act_s = actn_v.totok(act)
			if act_s[-1] == "(":
				if act_s == "SDRS(":
					stack.append([sdrs_idx,[]])
					sdrs_idx += 1
				elif re.match("^K[0-9]+\($", act_s):
					stack.append([1000+int(act_s[1:-1])-1, []])
				else:
					stack.append([-1,[]])
			elif actn_v.totok(act) == ")":
				b = stack.pop()
				if b[0] != -1 and b[0] < 1000:
					k_scope[b[0]] = b[1]
				if len(stack) > 0:
					stack[-1][1] = stack[-1][1] + b[1]
				if b[0] >= 100:
					stack[-1][1].append(b[0]%1000)
		sdrs_idx = 0
		# print k_scope
		# p_max
		p_max = -1
		for act in action_step1[1:]:
			if re.match("^P[0-9]+\($", actn_v.totok(act)):
				p_max = max(p_max, int(actn_v.totok(act)[1:-1])-1)

		cstns3.reset(p_max)
		for act in action_step1[1:]:
			cstn = cstns1.get_step_mask()
			#cstns._print_state()
			
			#print [actn_v.totok(x) for x in action[0][1:]]
			#print processed_act
			#print "required", act, actn_v.totok(act)
			#print "visible",
			#for i in range(len(cstn)):
			#	if cstn[i] == 1:
			#		print i,
			#print 
			#processed_act.append(actn_v.totok(act))
			assert cstn[act] == 1
			cstns1.update(act)
			if act in [actn_v.toidx("DRS("), actn_v.toidx("SDRS(")]:
				cstns2.reset_condition(act)
				if act == actn_v.toidx("SDRS("):
					cstns3.reset_condition(act, k_scope[sdrs_idx])
					sdrs_idx += 1
				else:
					cstns3.reset_condition(act)
				#print act
				for a in action_step2[idx]:
					cstn = cstns2.get_step_mask()
					#cstns2._print_state()

					if type(a) == types.StringType:
						#print a
						assert cstn[int(a[1:-1])+actn_v.size()] == 1
					else:
						#print actn_v.totok(a)
						assert cstn[a] == 1
					cstns2.update(a)

					if type(a) == types.StringType or a != actn_v.toidx(")"):
						if type(a) == types.StringType:
							a = int(a[1:-1])+actn_v.size()
						cstns3.reset_relation(a)
						for v in action_step3[idx2]:
							cstn = cstns3.get_step_mask()
							#cstns3._print_state()
							#print "required", v, actn_v.totok(v)
							#print "visible",
							#for i in range(len(cstn)):
							#	if cstn[i] == 1:
							#		print i,
							#print
							assert cstn[v] == 1
							cstns3.update(v)
						idx2 += 1
				idx += 1

def assign_hypers(subparser, hypers):
	for key in hypers.keys():
		if key[-3:] == "dim" or key[-5:] == "layer" or key[-2:] == "-l":
			subparser.add_argument("--"+key, default=int(hypers[key]))
		elif key[-4:] == "prob" or key[-2:] == "-f":
			subparser.add_argument("--"+key, default=float(hypers[key]))
		else:
			subparser.add_argument("--"+key, default=str(hypers[key]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()

	hypers = {}
	for line in open("DRS_config"):
		line = line.strip()
		if line == "" or line[0] == "#":
			continue
		hypers[line.split()[0]] = line.split()[1]

	subparser = subparsers.add_parser("train")
	subparser.set_defaults(callback=lambda args: run_train(args))
	assign_hypers(subparser, hypers)
	subparser.add_argument("--numpy-seed", type=int)
	subparser.add_argument("--model-path-base", required=True)
	subparser.add_argument("--dev-output-path-base", required=True)
	subparser.add_argument("--train-input", default="data/02-21.input")
	subparser.add_argument("--train-action", default="data/02-21.action")
	subparser.add_argument("--dev-input", default="data/22.input")
	subparser.add_argument("--dev-action", default="data/22.gold")
	subparser.add_argument("--dev-output", default="data/22.auto.clean.notop")
	subparser.add_argument("--batch-size", type=int, default=250)
	subparser.add_argument("--check-per-update", type=int, default=1000)
	subparser.add_argument("--eval-per-update", type=int, default=30000)
	subparser.add_argument("--eval-path-base", default="EVALB")
	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
	subparser.add_argument("--use-char", action='store_true')
	subparser.add_argument("--pretrain-path")
	subparser.add_argument("--action-dict-path", required=True)
	subparser.add_argument("--gpu", action='store_true')
	subparser.add_argument("--optimizer", default="adam")

	
	subparser = subparsers.add_parser("test")
	subparser.set_defaults(callback=lambda args: run_test(args))
	assign_hypers(subparser, hypers)
	subparser.add_argument("--model-path-base", required=True)
	subparser.add_argument("--test-output", required=True)
	subparser.add_argument("--test-input", required=True)
	subparser.add_argument("--pretrain-path")
	subparser.add_argument("--action-dict-path", required=True)
	subparser.add_argument("--use-char", action='store_true')
	subparser.add_argument("--gpu", action='store_true')
	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")


	subparser = subparsers.add_parser("check")
	subparser.set_defaults(callback=lambda args: run_check(args))
	assign_hypers(subparser, hypers)
	subparser.add_argument("--train-input", default="data/02-21.input")
	subparser.add_argument("--train-action", default="data/02-21.action")
	subparser.add_argument("--action-dict-path", required=True)

	args = parser.parse_args()
	args.callback(args)
