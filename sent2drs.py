import argparse
from system import system_check_and_init
from utils import read_input
from utils import read_input_test
from utils import get_singleton_dict
from utils import input2instance
from utils import read_tree
from utils import tree2action as tree2action
from utils import get_same_lemma

from dictionary.vocabulary import vocabulary
from dictionary.PretrainedEmb import PretrainedEmb
from representation.sentence_rep import sentence_rep

from encoder.bilstm import encoder_srnn as enc 
from decoder.lstm import decoder as dec

from utils import get_k_scope
from utils import get_p_max
from constraints.constraints import struct_constraints
from constraints.constraints import struct_constraints_state
from constraints.constraints import relation_constraints
from constraints.constraints import relation_constraints_state
from constraints.constraints import variable_constraints
from constraints.constraints import variable_constraints_state

import torch
from optimizer import optimizer

import types
import sys

# def run_train(args):
# 	system_check_and_init(args)
# 	if args.gpu:
# 		print "GPU available"
# 	else:
# 		print "CPU only"

# 	word_v = vocabulary()
# 	char_v = vocabulary()
# 	actn_v = vocabulary(UNK=False)
# 	pretrain = PretrainedEmb(args.pretrain_path)

# 	actn_v.toidx("<START>")
# 	actn_v.toidx("<END>")
# 	for i in range(args.X_l):
# 		actn_v.toidx("X"+str(i+1))
# 	for i in range(args.E_l):
# 		actn_v.toidx("E"+str(i+1))
# 	for i in range(args.S_l):
# 		actn_v.toidx("S"+str(i+1))
# 	for i in range(args.T_l):
# 		actn_v.toidx("T"+str(i+1))
# 	for i in range(args.P_l):
# 		actn_v.toidx("P"+str(i+1))
# 	for i in range(args.K_l):
# 		actn_v.toidx("K"+str(i+1))
# 	for i in range(args.P_l):
# 		actn_v.toidx("P"+str(i+1)+"(")
# 	for i in range(args.K_l):
# 		actn_v.toidx("K"+str(i+1)+"(")
# 	actn_v.toidx("CARD_NUMBER")
# 	actn_v.toidx("TIME_NUMBER")
# 	actn_v.toidx(")")

# 	#print actn_v.size()
# 	actn_v.read_file(args.action_dict_path)
# 	#print actn_v.size()
# 	actn_v.freeze()
# 	#instances
# 	train_input, train_sep = read_input(args.train_input)
# 	#print train_input[0]
# 	train_comb = [ get_same_lemma(x) for x in zip(train_input, train_sep)]
# 	#dev_input = read_input(args.dev_input)
# 	#dev_comb = [ get_same_lemma(x[1]) for x in dev_input]

# 	singleton_idx_dict, word_dict, word_v = get_singleton_dict(train_input, word_v)
# 	extra_vl = [ vocabulary() for i in range(len(train_input[0])-1)]	
# 	train_instance, word_v, char_v, extra_vl = input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, "train")
# 	word_v.freeze()
# 	char_v.freeze()
# 	for i in range(len(extra_vl)):
# 		extra_vl[i].freeze()
# 	#dev_instance, word_v, char_v, extra_vl = input2instance(dev_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")

# 	train_output = read_tree(args.train_action)
# 	#print train_output[0]
# 	#dev_output = read_output(args.dev_action)
# 	train_action = tree2action(train_output, actn_v)
# 	#dev_actoin, actn_v = output2action(dev_output, actn_v)
# 	#print train_action[0][0]
# 	#print train_action[0][1]
# 	#print train_action[0][2]
# 	print "word vocabulary size:", word_v.size()
# 	word_v.dump(args.model_path_base+"/word.list")
# 	print "char vocabulary size:", char_v.size()
# 	if args.use_char:
# 		char_v.dump(args.model_path_base+"/char.list")
# 	print "pretrain vocabulary size:", pretrain.size()
# 	extra_vl_size = []
# 	for i in range(len(extra_vl)):
# 		print "extra", i, "vocabulary size:", extra_vl[i].size()
# 		extra_vl[i].dump(args.model_path_base+"/extra."+str(i+1)+".list")
# 		extra_vl_size.append(extra_vl[i].size())
# 	print "action vocaluary size:", actn_v.size()
	
# 	#actn_v.dump()

# 	# neural components
# 	input_representation = sentence_rep(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
# 	encoder = None
# 	#if args.encoder == "BILSTM":
# 	#	from encoder.bilstm import encoder as enc
# 	#elif args.encoder == "Transformer":
# 	#	from encoder.transformer import encoder as enc
# 	encoder = enc(args)
# 	assert encoder, "please specify encoder type"
	
# 	#check dict to get index
# 	#BOX DISCOURSE RELATION PREDICATE CONSTANT
# 	starts = []
# 	ends = []
# 	lines = []
# 	for line in open(args.action_dict_path):
# 		line = line.strip()
# 		if line == "###":
# 			starts.append(actn_v.toidx(lines[0]))
# 			ends.append(actn_v.toidx(lines[-1]))
# 			lines = []
# 		else:
# 			if line[0] == "#":
# 				continue
# 			lines.append(line)

# 	#mask = Mask(args, actn_v, starts, ends)
# 	cstn_step1 = struct_constraints(actn_v, args)
# 	cstn_step2 = relation_constraints(actn_v, args, starts, ends)
# 	cstn_step3 = variable_constraints(actn_v, args)
# 	decoder = dec(actn_v.size(), args, actn_v, [cstn_step1, cstn_step2, cstn_step3])

# 	if args.gpu:
# 		encoder = encoder.cuda()
# 		decoder = decoder.cuda()
# 		input_representation = input_representation.cuda()
	
# 	#training process
# 	model_parameters = list(encoder.parameters()) + list(decoder.parameters()) + list(input_representation.parameters())
	
# 	model_optimizer = optimizer(args, model_parameters)
# 	lr = args.learning_rate_f

# 	i = len(train_instance)
# 	check_iter = 0
# 	check_loss1 = 0
# 	check_loss2 = 0
# 	check_loss3 = 0
# 	check_loss1_p = 0
# 	check_loss2_p = 0
# 	check_loss3_p = 0
# 	bscore = -1
# 	epoch = -1
# 	while True:
# 		"""
# 		for p in model_parameters:
# 			if p.grad is not None:
# 				p.grad.detach_()
# 				p.grad.zero_()
# 		"""
# 		model_optimizer.zero_grad()
# 		if i == len(train_instance):
# 			i = 0
# 			epoch += 1
# 			lr = args.learning_rate_f / (1 + epoch * args.learning_rate_decay_f)

# 		check_iter += 1
# 		input_t = input_representation(train_instance[i], singleton_idx_dict=singleton_idx_dict, train=True)
# 		word_rep_t, sent_rep_t, copy_rep_t, hidden_t = encoder(input_t, train_comb[i], train_sep[i], train=True)
# 		#step 1
# 		hidden_step1 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
# 		loss_t1, loss_p_t1, hidden_rep_t, hidden_step1 = decoder(train_action[i][0], hidden_step1, word_rep_t, sent_rep_t, pointer=train_action[i][3], copy_rep_t=None, train=True, state=None, opt=1)
# 		check_loss1 += loss_t1.data.tolist()
# 		check_loss1_p += loss_p_t1.data.tolist()
		
# 		#step 2
# 		idx = 0
# 		hidden_step2 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
# 		train_action_step2 = []
# 		#train_pointer_step2 = []
# 		for j in range(len(train_action[i][0])): #<START> DRS( P1(
# 			tok = train_action[i][0][j]
# 			if actn_v.totok(tok) == "DRS(":
# 				train_action_step2.append([hidden_rep_t[j], train_action[i][1][idx], train_action[i][4][idx][0]])
# 				#train_pointer_step2 += train_action[i][4][idx]
# 				idx += 1
# 			elif actn_v.totok(tok) == "SDRS(":
# 				train_action_step2.append([hidden_rep_t[j], train_action[i][1][idx], -1])
# 				idx += 1

# 		assert idx == len(train_action[i][1])
# 		loss_t2, hidden_rep_t, hidden_step2 = decoder(train_action_step2, hidden_step2, word_rep_t, sent_rep_t, pointer=None, copy_rep_t=copy_rep_t, train=True, state=None, opt=2)
# 		check_loss2 += loss_t2.data.tolist()
# 		#check_loss2_p += loss_p_t2.data.tolist()
# 		#step 3
# 		flat_train_action = [0] # <START>
# 		for l in train_action[i][1]:
# 			flat_train_action += l
# 		#print flat_train_action
# 		idx = 0
# 		hidden_step3 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
# 		train_action_step3 = []
# 		#train_pointer_step3 = []
# 		for j in range(len(flat_train_action)):
# 			tok = flat_train_action[j]
# 			#print tok
# 			if (type(tok) == types.StringType and tok[-1] == "(") or actn_v.totok(tok)[-1] == "(":
# 				train_action_step3.append([hidden_rep_t[j], train_action[i][2][idx]])
# 				#train_pointer_step3 += train_action[i][5][idx]
# 				idx += 1
# 		assert idx == len(train_action[i][2])
# 		loss_t3, hidden_rep_t, hidden_step3 = decoder(train_action_step3, hidden_step3, word_rep_t, sent_rep_t, pointer=None, copy_rep_t=None, train=True, state=None, opt=3)
# 		check_loss3 += loss_t3.data.tolist()
# 		#check_loss3_p += loss_p_t3.data.tolist()

# 		if check_iter % args.check_per_update == 0:
# 			print('epoch %.3f : structure %.5f, relation %.5f, variable %.5f, str_p %.5f' % (check_iter*1.0/len(train_instance), check_loss1*1.0 / args.check_per_update, check_loss2*1.0 / args.check_per_update, check_loss3*1.0 / args.check_per_update, check_loss1_p*1.0 / args.check_per_update))
# 			check_loss1 = 0
# 			check_loss2 = 0
# 			check_loss3 = 0

# 			check_loss1_p = 0
# 			#check_loss2_p = 0
# 			#check_loss3_p = 0
		
# 		i += 1
# 		loss_t = loss_t1 + loss_t2 + loss_t3 + loss_p_t1
# 		loss_t.backward()
# 		torch.nn.utils.clip_grad_value_(model_parameters, 5)

# 		#model_optimizer.step()
# 		"""
# 		for p in model_parameters:
# 			if p.requires_grad:
# 				p.data.add_(-lr, p.grad.data)
# 		"""
# 		model_optimizer.step()
		
# 		if check_iter % args.eval_per_update == 0:
# 			torch.save({"encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "input_representation": input_representation.state_dict()}, args.model_path_base+"/model"+str(int(check_iter/args.eval_per_update)))
# 			#test(args, args.dev_output, dev_instance, dev_comb, actn_v, input_representation, encoder, decoder)
		


# def test(args, output_file, test_instance, test_sep, test_comb, test_input, actn_v, input_representation, encoder, decoder):
	
# 	state_step1 = struct_constraints_state()
# 	state_step2 = relation_constraints_state()
# 	state_step3 = variable_constraints_state()
# 	test_outputs = []
# 	with open(output_file, "w") as w:
# 		for j, instance in enumerate(test_instance):
# 			print j
# 			test_input_t = input_representation(instance, singleton_idx_dict=None, train=False)
# 			test_word_rep_t, test_sent_rep_t, test_copy_rep_t, test_hidden_t= encoder(test_input_t, test_comb[j], test_sep[j], train=False)

# 			#step 1
# 			test_hidden_step1 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
# 			state_step1.reset()
# 			test_output_step1, test_pointers_step1, test_hidden_rep_step1, test_hidden_step1 = decoder(actn_v.toidx("<START>"), test_hidden_step1, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=None, train=False, state=state_step1, opt=1)
# 			#print test_output_step1	
# 			#print [actn_v.totok(x) for x in test_output_step1]
# 			#print test_hidden_rep_step1
# 			#print len(test_hidden_rep_step1)
# 			#exit(1)
# 			#step 2
# 			test_output_step2 = []
# 			test_hidden_rep_step2 = []
# 			test_hidden_step2 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			
# 			state_step2.reset() # <s> </s>
# 			drs_idx = 0
# 			for k in range(len(test_output_step1)): # DRS( P1(
# 				act1 = test_output_step1[k]
# 				act2 = None
# 				if k + 1 < len(test_output_step1):
# 					act2 = test_output_step1[k+1]
# 				if actn_v.totok(act1) == "DRS(":
# 					#print "DRS",test_pointers_step1[drs_idx]
# 					state_step2.reset_length(test_copy_rep_t[test_pointers_step1[drs_idx]].size(0))
# 					state_step2.reset_condition(act1, act2)
# 					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2, partial_state = decoder([test_hidden_rep_step1[k],test_pointers_step1[drs_idx]], test_hidden_step2, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=test_copy_rep_t, train=False, state=state_step2, opt=2)
# 					test_output_step2.append(one_test_output_step2)
# 					test_hidden_rep_step2.append(one_test_hidden_rep_step2)
# 					#partial_state is to store how many relation it already has
# 					state_step2.rel_g, state_step2.d_rel_g = partial_state
# 					drs_idx += 1
# 				if actn_v.totok(act1) == "SDRS(":
# 					#print "SDRS"
# 					state_step2.reset_length(0)
# 					state_step2.reset_condition(act1, act2)
# 					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2, partial_state = decoder([test_hidden_rep_step1[k],-1], test_hidden_step2, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=test_copy_rep_t, train=False, state=state_step2, opt=2)
# 					test_output_step2.append(one_test_output_step2)
# 					test_hidden_rep_step2.append(one_test_hidden_rep_step2)
# 					#partial_state is to store how many relation it already has
# 					state_step2.rel_g, state_step2.d_rel_g = partial_state
					
# 					#print test_hidden_step2

# 					#print one_test_hidden_rep_step2
# 					#print test_hidden_step2
# 					#exit(1)
# 			#print test_output_step2
# 			#step 3
# 			k_scope = get_k_scope(test_output_step1, actn_v)
# 			p_max = get_p_max(test_output_step1, actn_v)
			
# 			test_output_step3 = []
# 			test_hidden_step3 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
# 			state_step3.reset(p_max)
# 			k = 0
# 			sdrs_idx = 0
# 			for act1 in test_output_step1:
# 				if actn_v.totok(act1) in ["DRS(", "SDRS("]:
# 					if actn_v.totok(act1) == "SDRS(":
# 						state_step3.reset_condition(act1, k_scope[sdrs_idx])
# 						sdrs_idx += 1
# 					else:
# 						state_step3.reset_condition(act1)
# 					for kk in range(len(test_output_step2[k])-1): # rel( rel( )
# 						act2 = test_output_step2[k][kk]
# 						#if act2 >= actn_v.size():
# 						#	print "$"+str(act2 - actn_v.size())+"("
# 						#else:
# 						#	print actn_v.totok(act2)
							
# 						state_step3.reset_relation(act2)
# 						#print test_hidden_rep_step2[k][kk]
# 						#print test_hidden_step3
# 						#print "========================="
# 						one_test_output_step3, _, test_hidden_step3, partial_state = decoder(test_hidden_rep_step2[k][kk], test_hidden_step3, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=None, train=False, state=state_step3, opt=3)
# 						test_output_step3.append(one_test_output_step3)
# 						#partial state is to store how many variable it already has
# 						state_step3.x, state_step3.e, state_step3.s, state_step3.t = partial_state
# 						#exit(1)
# 					k += 1
# 			#print test_output_step3
# 			# write file
# 			test_output = []
# 			k = 0
# 			kk = 0
# 			drs_idx = 0
# 			for act1 in test_output_step1:
# 				if actn_v.totok(act1) == "DRS(":
# 					assert drs_idx < len(test_pointers_step1)
# 					test_output.append("DRS-"+str(test_pointers_step1[drs_idx])+"(")
# 					drs_idx += 1
# 				else:
# 					test_output.append(actn_v.totok(act1))
# 				if actn_v.totok(act1) in ["DRS(", "SDRS("]:
# 					for act2 in test_output_step2[k][:-1]:
# 						if act2 >= actn_v.size():
# 							test_output.append("$"+str(act2-actn_v.size())+"(")
# 						else:
# 							test_output.append(actn_v.totok(act2))
# 						for act3 in test_output_step3[kk]:
# 							test_output.append(actn_v.totok(act3))
# 						kk += 1
# 					k += 1
# 			w.write(out_tree(test_input[j][1], test_output)+"\n")
# 			w.flush()
# 			assert drs_idx == len(test_pointers_step1)
# 		w.close()

# def run_test(args):
# 	if args.soft_const:
# 		args.const = True
# 	word_v = vocabulary()
# 	word_v.read_file(args.model_path_base+"/word.list")
# 	word_v.freeze()

# 	char_v = vocabulary()
# 	if args.use_char:
# 		char_v.read_file(args.model_path_base+"/char.list")
# 		char_v.freeze()

# 	actn_v = vocabulary(UNK=False)
# 	pretrain = PretrainedEmb(args.pretrain_path)

# 	actn_v.toidx("<START>")
# 	actn_v.toidx("<END>")
# 	for i in range(args.X_l):
# 		actn_v.toidx("X"+str(i+1))
# 	for i in range(args.E_l):
# 		actn_v.toidx("E"+str(i+1))
# 	for i in range(args.S_l):
# 		actn_v.toidx("S"+str(i+1))
# 	for i in range(args.T_l):
# 		actn_v.toidx("T"+str(i+1))
# 	for i in range(args.P_l):
# 		actn_v.toidx("P"+str(i+1))
# 	for i in range(args.K_l):
# 		actn_v.toidx("K"+str(i+1))
# 	for i in range(args.P_l):
# 		actn_v.toidx("P"+str(i+1)+"(")
# 	for i in range(args.K_l):
# 		actn_v.toidx("K"+str(i+1)+"(")
# 	actn_v.toidx("CARD_NUMBER")
# 	actn_v.toidx("TIME_NUMBER")
# 	actn_v.toidx(")")

# 	actn_v.read_file(args.action_dict_path)
# 	actn_v.freeze()

# 	test_input, test_sep = read_input_test(args.test_input)
# 	test_comb = [ get_same_lemma(x) for x in zip(test_input,test_sep)]
# 	extra_vl = [ vocabulary() for i in range(len(test_input[0])-1)]
# 	for i in range(len(test_input[0])-1):
# 		extra_vl[i].read_file(args.model_path_base+"/extra."+str(i+1)+".list")
# 		extra_vl[i].freeze()

# 	print "word vocabulary size:", word_v.size()
# 	print "char vocabulary size:", char_v.size() 
# 	print "pretrain vocabulary size:", pretrain.size()
# 	extra_vl_size = []
# 	for i in range(len(extra_vl)):
# 		print "extra", i, "vocabulary size:", extra_vl[i].size()
# 		extra_vl_size.append(extra_vl[i].size())
# 	print "action vocaluary size:", actn_v.size() 

# 	input_representation = sentence_rep(word_v.size(), char_v.size(), pretrain, extra_vl_size, args)
# 	encoder = None
# 	#if args.encoder == "BILSTM":
# 	#	from encoder.bilstm import encoder as enc
# 	#elif args.encoder == "Transformer":
# 	#	from encoder.transformer import encoder as enc
# 	encoder = enc(args)
# 	assert encoder, "please specify encoder type"
	
# 	#check dict to get index
# 	#BOX DISCOURSE RELATION PREDICATE CONSTANT
# 	starts = []
# 	ends = []
# 	lines = []
# 	for line in open(args.action_dict_path):
# 		line = line.strip()
# 		if line == "###":
# 			starts.append(actn_v.toidx(lines[0]))
# 			ends.append(actn_v.toidx(lines[-1]))
# 			lines = []
# 		else:
# 			if line[0] == "#":
# 				continue
# 			lines.append(line)

# 	#mask = Mask(args, actn_v, starts, ends)
# 	cstn_step1 = struct_constraints(actn_v, args)
# 	cstn_step2 = relation_constraints(actn_v, args, starts, ends)
# 	cstn_step3 = variable_constraints(actn_v, args)
# 	decoder = dec(actn_v.size(), args, actn_v, [cstn_step1, cstn_step2, cstn_step3])

# 	check_point = torch.load(args.model_path_base+"/model")
# 	encoder.load_state_dict(check_point["encoder"])
# 	decoder.load_state_dict(check_point["decoder"])
# 	input_representation.load_state_dict(check_point["input_representation"])

# 	if args.gpu:
# 		encoder = encoder.cuda()
# 		decoder = decoder.cuda()
# 		input_representation = input_representation.cuda()
	
# 	test_instance, word_v, char_v, extra_vl = input2instance(test_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")
	
# 	test(args, args.test_output, test_instance, test_sep, test_comb, test_input, actn_v, input_representation, encoder, decoder)

# def out_tree(lemmas, trees):
# 	lems = " ".join(lemmas[1:-1]).split("|||")
# 	for i,x in enumerate(lems):
# 		lems[i] = []
# 		for y in x.split():
# 			if y not in lems[i]:
# 				lems[i].append(y)
# 	cur = 0
# 	j = 0
# 	while j < len(trees):
# 		if re.match("^DRS-[0-9]+\($", trees[j]):
# 			cur = int(trees[j][4:-1])
# 			assert cur < len(lems)
# 		elif re.match("^\$[0-9]+\(", trees[j]):
# 			idx = int(trees[j][1:-1])
# 			trees[j] = lems[cur][idx]+"("
# 		j += 1
# 	return " ".join(trees)

	
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import sent_tokenize
import re

class Demo:
	def __init__(self):
		pass

	def load_model(self, args):
		self.args = args
		if args.soft_const:
			args.const = True
		self.word_v = vocabulary()
		self.word_v.read_file(args.model_path_base+"/word.list")
		self.word_v.freeze()

		self.char_v = vocabulary()
		if args.use_char:
			self.char_v.read_file(args.model_path_base+"/char.list")
			self.char_v.freeze()

		self.actn_v = vocabulary(UNK=False)
		self.pretrain = PretrainedEmb(args.pretrain_path)

		self.actn_v.toidx("<START>")
		self.actn_v.toidx("<END>")
		for i in range(args.X_l):
			self.actn_v.toidx("X"+str(i+1))
		for i in range(args.E_l):
			self.actn_v.toidx("E"+str(i+1))
		for i in range(args.S_l):
			self.actn_v.toidx("S"+str(i+1))
		for i in range(args.T_l):
			self.actn_v.toidx("T"+str(i+1))
		for i in range(args.P_l):
			self.actn_v.toidx("P"+str(i+1))
		for i in range(args.K_l):
			self.actn_v.toidx("K"+str(i+1))
		for i in range(args.P_l):
			self.actn_v.toidx("P"+str(i+1)+"(")
		for i in range(args.K_l):
			self.actn_v.toidx("K"+str(i+1)+"(")
		self.actn_v.toidx("CARD_NUMBER")
		self.actn_v.toidx("TIME_NUMBER")
		self.actn_v.toidx(")")

		self.actn_v.read_file(args.action_dict_path)
		self.actn_v.freeze()

		self.extra_vl = [ vocabulary() for i in range(1)]
		for i in range(1):
			self.extra_vl[i].read_file(args.model_path_base+"/extra."+str(i+1)+".list")
			self.extra_vl[i].freeze()

		print "word vocabulary size:", self.word_v.size()
		print "char vocabulary size:", self.char_v.size() 
		print "pretrain vocabulary size:", self.pretrain.size()
		self.extra_vl_size = []
		for i in range(len(self.extra_vl)):
			print "extra", i, "vocabulary size:", self.extra_vl[i].size()
			self.extra_vl_size.append(self.extra_vl[i].size())
		print "action vocaluary size:", self.actn_v.size() 

		self.input_representation = sentence_rep(self.word_v.size(), self.char_v.size(), self.pretrain, self.extra_vl_size, self.args)
		self.encoder = None
		#if args.encoder == "BILSTM":
		#	from encoder.bilstm import encoder as enc
		#elif args.encoder == "Transformer":
		#	from encoder.transformer import encoder as enc
		self.encoder = enc(args)
		assert self.encoder, "please specify encoder type"
	
		#check dict to get index
		#BOX DISCOURSE RELATION PREDICATE CONSTANT
		starts = []
		ends = []
		lines = []
		for line in open(args.action_dict_path):
			line = line.strip()
			if line == "###":
				starts.append(self.actn_v.toidx(lines[0]))
				ends.append(self.actn_v.toidx(lines[-1]))
				lines = []
			else:
				if line[0] == "#":
					continue
				lines.append(line)

		#mask = Mask(args, actn_v, starts, ends)
		self.cstn_step1 = struct_constraints(self.actn_v, args)
		self.cstn_step2 = relation_constraints(self.actn_v, args, starts, ends)
		self.cstn_step3 = variable_constraints(self.actn_v, args)
		self.decoder = dec(self.actn_v.size(), self.args, self.actn_v, [self.cstn_step1, self.cstn_step2, self.cstn_step3])


		check_point = torch.load(args.model_path_base+"/model", map_location=lambda storage, loc: storage)
		self.encoder.load_state_dict(check_point["encoder"])
		self.decoder.load_state_dict(check_point["decoder"])
		self.input_representation.load_state_dict(check_point["input_representation"])

		if args.gpu:
			self.encoder = self.encoder.cuda()
			self.decoder = self.decoder.cuda()
			self.input_representation = self.input_representation.cuda()
	
	def get_wordnet_pos(self,treebank_tag):
		"""
		return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
		"""
		if treebank_tag.startswith('J'):
			return wordnet.ADJ
		elif treebank_tag.startswith('V'):
			return wordnet.VERB
		elif treebank_tag.startswith('N'):
			return wordnet.NOUN
		elif treebank_tag.startswith('R'):
			return wordnet.ADV
		else:
			# As default pos in lemmatization is Noun
			return wordnet.NOUN
	def get_lemmas(self, tokens):
		lemmatizer = WordNetLemmatizer()
		pos_tokens = [nltk.pos_tag(tokens)]
		lemmas = []
		for pos in pos_tokens[0]:
			word, pos_tag = pos
			lemmas.append(lemmatizer.lemmatize(word.lower(),self.get_wordnet_pos(pos_tag)))
			lemmas[-1] = lemmas[-1].encode("utf8")
		return lemmas

	def out_tree(self, lemmas, trees):
		# for one instance
		lems = " ".join(lemmas[1:-1]).split("|||")
		for i,x in enumerate(lems):
			lems[i] = x.split()
		
		cur = 0
		j = 0
		while j < len(trees):
			if re.match("^DRS-[0-9]+\($", trees[j]):
				cur = int(trees[j][4:-1])
				trees[j] = "DRS("
				assert cur < len(lems[i])
			elif re.match("^\$[0-9]+\(", trees[j]):
				idx = int(trees[j][1:-1])
				trees[j] = lems[cur][idx]+"("
			j += 1
		# print " ".join(trees)

		return " ".join(trees)


	def test(self, docs):
		docs = sent_tokenize(docs)
		tokenizer = nltk.tokenize.TreebankWordTokenizer()

		doc_toks = ""
		doc_lems = ""
		for sent in docs:
			tokens = tokenizer.tokenize(sent)
			lemmas = self.get_lemmas(tokens)
			doc_toks += " ".join(tokens) + " ||| "
			doc_lems += " ".join(lemmas) + " ||| "
		doc_toks = ["<s>"] + doc_toks[:-5].split() + ["</s>"]
		doc_lems = ["<s>"] + doc_lems[:-5].split() + ["</s>"]
		
		test_input = [[doc_toks, doc_lems]]
		test_sep = [[]]
		for i, t in enumerate(doc_toks):
			if t in ["<s>", "</s>", "|||"]:
				test_sep[-1].append(i)	
		test_comb = [ get_same_lemma(x) for x in zip(test_input, test_sep)]
		test_instance, _, _, _ = input2instance(test_input, self.word_v, self.char_v, self.pretrain, self.extra_vl, {}, self.args, "dev")

		test_output = self.decode(test_instance, test_sep, test_comb)
		return self.out_tree(doc_lems, test_output)
	def decode(self, test_instance, test_sep, test_comb):
	
		state_step1 = struct_constraints_state()
		state_step2 = relation_constraints_state()
		state_step3 = variable_constraints_state()
		for j, instance in enumerate(test_instance):
			test_input_t = self.input_representation(instance, singleton_idx_dict=None, train=False)
			test_word_rep_t, test_sent_rep_t, test_copy_rep_t, test_hidden_t= self.encoder(test_input_t, test_comb[j], test_sep[j], train=False)

			#step 1
			test_hidden_step1 = (test_hidden_t[0].view(self.args.action_n_layer, 1, -1), test_hidden_t[1].view(self.args.action_n_layer, 1, -1))
			state_step1.reset()
			test_output_step1, test_pointers_step1, test_hidden_rep_step1, test_hidden_step1 = self.decoder(self.actn_v.toidx("<START>"), test_hidden_step1, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=None, train=False, state=state_step1, opt=1)
			#print test_output_step1	
			#print [actn_v.totok(x) for x in test_output_step1]
			#print test_hidden_rep_step1
			#print len(test_hidden_rep_step1)
			#exit(1)
			#step 2
			test_output_step2 = []
			test_hidden_rep_step2 = []
			test_hidden_step2 = (test_hidden_t[0].view(self.args.action_n_layer, 1, -1), test_hidden_t[1].view(self.args.action_n_layer, 1, -1))
			
			state_step2.reset() # <s> </s>
			drs_idx = 0
			for k in range(len(test_output_step1)): # DRS( P1(
				act1 = test_output_step1[k]
				act2 = None
				if k + 1 < len(test_output_step1):
					act2 = test_output_step1[k+1]
				if self.actn_v.totok(act1) == "DRS(":
					#print "DRS",test_pointers_step1[drs_idx]
					state_step2.reset_length(test_copy_rep_t[test_pointers_step1[drs_idx]].size(0))
					state_step2.reset_condition(act1, act2)
					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2, partial_state = self.decoder([test_hidden_rep_step1[k],test_pointers_step1[drs_idx]], test_hidden_step2, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=test_copy_rep_t, train=False, state=state_step2, opt=2)
					test_output_step2.append(one_test_output_step2)
					test_hidden_rep_step2.append(one_test_hidden_rep_step2)
					#partial_state is to store how many relation it already has
					state_step2.rel_g, state_step2.d_rel_g = partial_state
					drs_idx += 1
				if self.actn_v.totok(act1) == "SDRS(":
					#print "SDRS"
					state_step2.reset_length(0)
					state_step2.reset_condition(act1, act2)
					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2, partial_state = self.decoder([test_hidden_rep_step1[k],-1], test_hidden_step2, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=test_copy_rep_t, train=False, state=state_step2, opt=2)
					test_output_step2.append(one_test_output_step2)
					test_hidden_rep_step2.append(one_test_hidden_rep_step2)
					#partial_state is to store how many relation it already has
					state_step2.rel_g, state_step2.d_rel_g = partial_state
					
					#print test_hidden_step2

					#print one_test_hidden_rep_step2
					#print test_hidden_step2
					#exit(1)
			#print test_output_step2
			#step 3
			k_scope = get_k_scope(test_output_step1, self.actn_v)
			p_max = get_p_max(test_output_step1, self.actn_v)
			
			test_output_step3 = []
			test_hidden_step3 = (test_hidden_t[0].view(self.args.action_n_layer, 1, -1), test_hidden_t[1].view(self.args.action_n_layer, 1, -1))
			state_step3.reset(p_max)
			k = 0
			sdrs_idx = 0
			for act1 in test_output_step1:
				if self.actn_v.totok(act1) in ["DRS(", "SDRS("]:
					if self.actn_v.totok(act1) == "SDRS(":
						state_step3.reset_condition(act1, k_scope[sdrs_idx])
						sdrs_idx += 1
					else:
						state_step3.reset_condition(act1)
					for kk in range(len(test_output_step2[k])-1): # rel( rel( )
						act2 = test_output_step2[k][kk]
						#if act2 >= actn_v.size():
						#	print "$"+str(act2 - actn_v.size())+"("
						#else:
						#	print actn_v.totok(act2)
							
						state_step3.reset_relation(act2)
						#print test_hidden_rep_step2[k][kk]
						#print test_hidden_step3
						#print "========================="
						one_test_output_step3, _, test_hidden_step3, partial_state = self.decoder(test_hidden_rep_step2[k][kk], test_hidden_step3, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=None, train=False, state=state_step3, opt=3)
						test_output_step3.append(one_test_output_step3)
						#partial state is to store how many variable it already has
						state_step3.x, state_step3.e, state_step3.s, state_step3.t = partial_state
						#exit(1)
					k += 1
			#print test_output_step3
			# write file
			test_output = []
			k = 0
			kk = 0
			drs_idx = 0
			for act1 in test_output_step1:
				if self.actn_v.totok(act1) == "DRS(":
					assert drs_idx < len(test_pointers_step1)
					test_output.append("DRS-"+str(test_pointers_step1[drs_idx])+"(")
					drs_idx += 1
				else:
					test_output.append(self.actn_v.totok(act1))
				if self.actn_v.totok(act1) in ["DRS(", "SDRS("]:
					for act2 in test_output_step2[k][:-1]:
						if act2 >= self.actn_v.size():
							test_output.append("$"+str(act2-self.actn_v.size())+"(")
						else:
							test_output.append(self.actn_v.totok(act2))
						for act3 in test_output_step3[kk]:
							test_output.append(self.actn_v.totok(act3))
						kk += 1
					k += 1
			assert drs_idx == len(test_pointers_step1)
			return test_output

def easy_use(args):
	demo = Demo(args)
	docs = ["The American Civil Liberties Union (ACLU), which filed a lawsuit seeking the release of the materials, hailed the decision as a step toward ensuring government leaders are held accountable for abuses that happened on their watch.", "British sprinter Dwain Chambers and his lawyers have appeared in London's High Court seeking a temporary injunction against his lifetime Olympic doping ban."]
	docs = ["John likes eating apples.", "So he bought a lot of apples."]
	demo.test(docs)


# def assign_hypers(subparser, hypers):
# 	for key in hypers.keys():
# 		if key[-3:] == "dim" or key[-5:] == "layer" or key[-2:] == "-l":
# 			subparser.add_argument("--"+key, default=int(hypers[key]))
# 		elif key[-4:] == "prob" or key[-2:] == "-f":
# 			subparser.add_argument("--"+key, default=float(hypers[key]))
# 		else:
# 			subparser.add_argument("--"+key, default=str(hypers[key]))

# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser()
# 	subparsers = parser.add_subparsers()

# 	hypers = {}
# 	for line in open("DRS_config"):
# 		line = line.strip()
# 		if line == "" or line[0] == "#":
# 			continue
# 		hypers[line.split()[0]] = line.split()[1]

# 	subparser = subparsers.add_parser("train")
# 	subparser.set_defaults(callback=lambda args: run_train(args))
# 	assign_hypers(subparser, hypers)
# 	subparser.add_argument("--numpy-seed", type=int)
# 	subparser.add_argument("--model-path-base", required=True)
# 	subparser.add_argument("--train-input", required=True)
# 	subparser.add_argument("--train-action", required=True)
# 	subparser.add_argument("--batch-size", type=int, default=1)
# 	subparser.add_argument("--beam-size", type=int, default=1)
# 	subparser.add_argument("--check-per-update", type=int, default=1000)
# 	subparser.add_argument("--eval-per-update", type=int, default=30000)
# 	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
# 	subparser.add_argument("--use-char", action='store_true')
# 	subparser.add_argument("--pretrain-path")
# 	subparser.add_argument("--action-dict-path", required=True)
# 	subparser.add_argument("--gpu", action='store_true')
# 	subparser.add_argument("--optimizer", default="adam")

	
# 	subparser = subparsers.add_parser("test")
# 	subparser.set_defaults(callback=lambda args: run_test(args))
# 	assign_hypers(subparser, hypers)
# 	subparser.add_argument("--model-path-base", required=True)
# 	subparser.add_argument("--test-output", required=True)
# 	subparser.add_argument("--test-input", required=True)
# 	subparser.add_argument("--pretrain-path")
# 	subparser.add_argument("--beam-size", type=int, default=1)
# 	subparser.add_argument("--action-dict-path", required=True)
# 	subparser.add_argument("--use-char", action='store_true')
# 	subparser.add_argument("--gpu", action='store_true')
# 	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
# 	subparser.add_argument("--const", action="store_true")
# 	subparser.add_argument("--soft-const", action="store_true")

# 	subparser = subparsers.add_parser("easy")
#         subparser.set_defaults(callback=lambda args: easy_use(args))
#         assign_hypers(subparser, hypers)
#         subparser.add_argument("--model-path-base", required=True)
#         subparser.add_argument("--test-output", required=True)
#         subparser.add_argument("--test-input", required=True)
#         subparser.add_argument("--pretrain-path")
#         subparser.add_argument("--beam-size", type=int, default=1)
#         subparser.add_argument("--action-dict-path", required=True)
#         subparser.add_argument("--use-char", action='store_true')
#         subparser.add_argument("--gpu", action='store_true')
#         subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
#         subparser.add_argument("--const", action="store_true")
#         subparser.add_argument("--soft-const", action="store_true")

# 	subparser = subparsers.add_parser("check")
# 	subparser.set_defaults(callback=lambda args: run_check(args))
# 	assign_hypers(subparser, hypers)
# 	subparser.add_argument("--train-input", default="data/02-21.input")
# 	subparser.add_argument("--train-action", default="data/02-21.action")
# 	subparser.add_argument("--action-dict-path", required=True)

# 	args = parser.parse_args()
# 	args.callback(args)