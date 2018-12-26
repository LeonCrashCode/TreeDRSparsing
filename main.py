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

from encoder.bilstm import encoder_srnn as enc 
from decoder.lstm import decoder as dec

from utils import get_k_scope
from utils import get_p_max
from utils import get_b_max
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
	#actn_v.toidx("@B")
	actn_v.toidx("B0")
	for i in range(args.B_l):
		actn_v.toidx("B"+str(i+1))
	#actn_v.toidx("@X")
	for i in range(args.X_l):
		actn_v.toidx("X"+str(i+1))
	#actn_v.toidx("@E")
	for i in range(args.E_l):
		actn_v.toidx("E"+str(i+1))
	#actn_v.toidx("@S")
	for i in range(args.S_l):
		actn_v.toidx("S"+str(i+1))
	#actn_v.toidx("@T")
	for i in range(args.T_l):
		actn_v.toidx("T"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1))
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1))
	actn_v.toidx("@P(")
	#for i in range(args.P_l):
		#actn_v.toidx("P"+str(i+1)+"(")
	actn_v.toidx("@K(")
	#for i in range(args.K_l):
		#actn_v.toidx("K"+str(i+1)+"(")
	actn_v.toidx("CARD_NUMBER")
	actn_v.toidx("TIME_NUMBER")
	actn_v.toidx(")")

	#print actn_v.size()
	actn_v.read_file(args.action_dict_path)
	#print actn_v.size()
	actn_v.freeze()
	#instances
	train_input, train_sep = read_input(args.train_input)
	#print train_input[0]
	#train_comb = [ get_same_lemma(x) for x in zip(train_input, train_sep)]
	#dev_input = read_input(args.dev_input)
	#dev_comb = [ get_same_lemma(x[1]) for x in dev_input]

	singleton_idx_dict, word_dict, word_v = get_singleton_dict(train_input, word_v)
	extra_vl = [ vocabulary() for i in range(len(train_input[0])-1)]	
	train_instance, word_v, char_v, extra_vl = input2instance(train_input, word_v, char_v, pretrain, extra_vl, word_dict, args, "train")
	#print train_instance[0]
	word_v.freeze()
	char_v.freeze()
	for i in range(len(extra_vl)):
		extra_vl[i].freeze()
	#dev_instance, word_v, char_v, extra_vl = input2instance(dev_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")

	train_output = read_tree(args.train_action)
	#print train_output[0]
	#dev_output = read_output(args.dev_action)
	train_action = tree2action(train_output, actn_v)
	#dev_actoin, actn_v = output2action(dev_output, actn_v)
	
	"""
	print train_action[0][0]
	for item in train_action[0][0]:
		if type(item) == types.StringType:
			print item,
		else:
			print actn_v.totok(item),
	print
	print train_action[0][1]
	for item1 in train_action[0][1]:
		for item in item1:
                	if type(item) == types.StringType:
                        	print item,
                	else:
                        	print actn_v.totok(item),
        	print
	print train_action[0][2]
	for item1 in train_action[0][2]:
		for item in item1:
                	if type(item) == types.StringType:
                        	print item,
                	else:
                        	print actn_v.totok(item),
        	print
	"""
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
	#BOX DISCOURSE RELATION PREDICATE SENSE
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
	cstn_step1 = struct_constraints(actn_v, args)
	cstn_step2 = relation_constraints(actn_v, args, starts, ends)
	cstn_step3 = variable_constraints(actn_v, args, starts[-1], ends[-1]) #SENSE
	decoder = dec(actn_v.size(), args, actn_v, [cstn_step1, cstn_step2, cstn_step3])

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
	check_loss1_p = 0
	check_loss2_p = 0
	check_loss3_p = 0
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
		word_rep_t, sent_rep_t, copy_rep_t, hidden_t = encoder(input_t, train_sep[i], train=True)
		#step 1
		hidden_step1 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
		loss_t1, loss_p_t1, hidden_rep_t, hidden_step1 = decoder(train_action[i][0], hidden_step1, word_rep_t, sent_rep_t, pointer=train_action[i][3], copy_rep_t=None, train=True, state=None, opt=1)
		check_loss1 += loss_t1.data.tolist()
		check_loss1_p += loss_p_t1.data.tolist()
		
		#step 2
		idx = 0
		hidden_step2 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
		train_action_step2 = []
		#train_pointer_step2 = []
		for j in range(len(train_action[i][0])): #<START> DRS( P1(
			tok = train_action[i][0][j]
			if actn_v.totok(tok) == "DRS(":
				train_action_step2.append([hidden_rep_t[j], train_action[i][1][idx], train_action[i][4][idx][0]])
				#train_pointer_step2 += train_action[i][4][idx]
				idx += 1
			elif actn_v.totok(tok) == "SDRS(":
				train_action_step2.append([hidden_rep_t[j], train_action[i][1][idx], -1])
				idx += 1

		assert idx == len(train_action[i][1])
		loss_t2, hidden_rep_t, hidden_step2 = decoder(train_action_step2, hidden_step2, word_rep_t, sent_rep_t, pointer=None, copy_rep_t=copy_rep_t, train=True, state=None, opt=2)
		check_loss2 += loss_t2.data.tolist()
		#check_loss2_p += loss_p_t2.data.tolist()
		#step 3
		flat_train_action = [0] # <START>
		for l in train_action[i][1]:
			flat_train_action += l
		#print flat_train_action
		idx = 0
		hidden_step3 = (hidden_t[0].view(args.action_n_layer, 1, -1), hidden_t[1].view(args.action_n_layer, 1, -1))
		train_action_step3 = []
		#train_pointer_step3 = []
		for j in range(len(flat_train_action)):
			tok = flat_train_action[j]
			#print tok
			if (type(tok) == types.StringType and tok[-1] == "(") or actn_v.totok(tok)[-1] == "(":
				if type(tok) != types.StringType and tok >= starts[1] and tok <= ends[1]:
					# is discourse relation
					train_action_step3.append([hidden_rep_t[j], train_action[i][2][idx], -1])
				else:
					train_action_step3.append([hidden_rep_t[j], train_action[i][2][idx], train_action[i][5][idx][0]])
					#train_pointer_step3 += train_action[i][5][idx]
				idx += 1
		assert idx == len(train_action[i][2])
		loss_t3, hidden_rep_t, hidden_step3 = decoder(train_action_step3, hidden_step3, word_rep_t, sent_rep_t, pointer=None, copy_rep_t=copy_rep_t, train=True, state=None, opt=3)
		check_loss3 += loss_t3.data.tolist()
		#check_loss3_p += loss_p_t3.data.tolist()
		if check_iter % args.check_per_update == 0:
			print('epoch %.3f : structure %.5f, relation %.5f, variable %.5f, str_p %.5f' % (check_iter*1.0/len(train_instance), check_loss1*1.0 / args.check_per_update, check_loss2*1.0 / args.check_per_update, check_loss3*1.0 / args.check_per_update, check_loss1_p*1.0 / args.check_per_update))
			check_loss1 = 0
			check_loss2 = 0
			check_loss3 = 0

			check_loss1_p = 0
			#check_loss2_p = 0
			#check_loss3_p = 0
		
		i += 1
		loss_t = loss_t1 + loss_t2 + loss_t3 + loss_p_t1
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
			#test(args, args.dev_output, dev_instance, dev_comb, actn_v, input_representation, encoder, decoder)
		


def test(args, output_file, test_instance, test_sep, test_comb, actn_v, input_representation, encoder, decoder):
	
	state_step1 = struct_constraints_state()
	state_step2 = relation_constraints_state()
	state_step3 = variable_constraints_state()
	with open(output_file, "w") as w:
		for j, instance in enumerate(test_instance):
			print j
			test_input_t = input_representation(instance, singleton_idx_dict=None, train=False)
			test_word_rep_t, test_sent_rep_t, test_copy_rep_t, test_hidden_t= encoder(test_input_t, test_comb[j], test_sep[j], train=False)

			#step 1
			test_hidden_step1 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			state_step1.reset()
			test_output_step1, test_pointers_step1, test_hidden_rep_step1, test_hidden_step1 = decoder(actn_v.toidx("<START>"), test_hidden_step1, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=None, train=False, state=state_step1, opt=1)
			#print test_output_step1	
			#print [actn_v.totok(x) for x in test_output_step1]
			#print test_hidden_rep_step1
			#print len(test_hidden_rep_step1)
			#exit(1)
			#step 2
			test_output_step2 = []
			test_hidden_rep_step2 = []
			test_hidden_step2 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			
			state_step2.reset() # <s> </s>
			drs_idx = 0
			for k in range(len(test_output_step1)): # DRS( P1(
				act1 = test_output_step1[k]
				act2 = None
				if k + 1 < len(test_output_step1):
					act2 = test_output_step1[k+1]
				if actn_v.totok(act1) == "DRS(":
					#print "DRS",test_pointers_step1[drs_idx]
					state_step2.reset_length(test_copy_rep_t[test_pointers_step1[drs_idx]].size(0))
					state_step2.reset_condition(act1, act2)
					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2, partial_state = decoder([test_hidden_rep_step1[k],test_pointers_step1[drs_idx]], test_hidden_step2, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=test_copy_rep_t, train=False, state=state_step2, opt=2)
					test_output_step2.append(one_test_output_step2)
					test_hidden_rep_step2.append(one_test_hidden_rep_step2)
					#partial_state is to store how many relation it already has
					state_step2.rel_g, state_step2.d_rel_g = partial_state
					drs_idx += 1
				if actn_v.totok(act1) == "SDRS(":
					#print "SDRS"
					state_step2.reset_length(0)
					state_step2.reset_condition(act1, act2)
					one_test_output_step2, one_test_hidden_rep_step2, test_hidden_step2, partial_state = decoder([test_hidden_rep_step1[k],-1], test_hidden_step2, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=test_copy_rep_t, train=False, state=state_step2, opt=2)
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
			k_scope = get_k_scope(test_output_step1, actn_v)
			p_max = get_p_max(test_output_step1, actn_v)
			
			test_output_step3 = []
			test_hidden_step3 = (test_hidden_t[0].view(args.action_n_layer, 1, -1), test_hidden_t[1].view(args.action_n_layer, 1, -1))
			state_step3.reset(p_max)
			k = 0
			sdrs_idx = 0
			for act1 in test_output_step1:
				if actn_v.totok(act1) in ["DRS(", "SDRS("]:
					if actn_v.totok(act1) == "SDRS(":
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
						one_test_output_step3, _, test_hidden_step3, partial_state = decoder(test_hidden_rep_step2[k][kk], test_hidden_step3, test_word_rep_t, test_sent_rep_t, pointer=None, copy_rep_t=None, train=False, state=state_step3, opt=3)
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
				if actn_v.totok(act1) == "DRS(":
					assert drs_idx < len(test_pointers_step1)
					test_output.append("DRS-"+str(test_pointers_step1[drs_idx])+"(")
					drs_idx += 1
				else:
					test_output.append(actn_v.totok(act1))
				if actn_v.totok(act1) in ["DRS(", "SDRS("]:
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
			assert drs_idx == len(test_pointers_step1)
		w.close()


def run_test(args):
	if args.soft_const:
		args.const = True
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

	test_input, test_sep = read_input(args.test_input)
	test_comb = [ get_same_lemma(x) for x in zip(test_input,test_sep)]
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
	cstn_step1 = struct_constraints(actn_v, args)
	cstn_step2 = relation_constraints(actn_v, args, starts, ends)
	cstn_step3 = variable_constraints(actn_v, args)
	decoder = dec(actn_v.size(), args, actn_v, [cstn_step1, cstn_step2, cstn_step3])

	check_point = torch.load(args.model_path_base+"/model")
	encoder.load_state_dict(check_point["encoder"])
	decoder.load_state_dict(check_point["decoder"])
	input_representation.load_state_dict(check_point["input_representation"])

	if args.gpu:
		encoder = encoder.cuda()
		decoder = decoder.cuda()
		input_representation = input_representation.cuda()
	
	test_instance, word_v, char_v, extra_vl = input2instance(test_input, word_v, char_v, pretrain, extra_vl, {}, args, "dev")
	
	test(args, args.test_output, test_instance, test_sep, test_comb, actn_v, input_representation, encoder, decoder)
	
def run_check(args):

	import re
	actn_v = vocabulary(UNK=False)
	actn_v.toidx("<START>")
	actn_v.toidx("<END>")
	#actn_v.toidx("@B")
	actn_v.toidx("B0")
	for i in range(args.B_l):
		actn_v.toidx("B"+str(i+1))
	#actn_v.toidx("@X")
	for i in range(args.X_l):
		actn_v.toidx("X"+str(i+1))
	#actn_v.toidx("@E")
	for i in range(args.E_l):
		actn_v.toidx("E"+str(i+1))
	#actn_v.toidx("@S")
	for i in range(args.S_l):
		actn_v.toidx("S"+str(i+1))
	#actn_v.toidx("@T")
	for i in range(args.T_l):
		actn_v.toidx("T"+str(i+1))
	for i in range(args.P_l):
		actn_v.toidx("P"+str(i+1))
	for i in range(args.K_l):
		actn_v.toidx("K"+str(i+1))
	actn_v.toidx("@P(")
	#for i in range(args.P_l):
		#actn_v.toidx("P"+str(i+1)+"(")
	actn_v.toidx("@K(")
	#for i in range(args.K_l):
		#actn_v.toidx("K"+str(i+1)+"(")
	actn_v.toidx("CARD_NUMBER")
	actn_v.toidx("TIME_NUMBER")
	actn_v.toidx(")")

	actn_v.read_file(args.action_dict_path)
	actn_v.freeze()
	
	train_input, train_sep = read_input(args.train_input)
	train_output = read_tree(args.train_action)
	train_action = tree2action(train_output, actn_v)

	#dev_actoin, actn_v = output2action(dev_output, actn_v)
	print "action vocaluary size:", actn_v.size()

	#check dict to get index
	#BOX DISCOURSE RELATION PREDICATE SENSE
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

	cstn_step1 = struct_constraints(actn_v, args)
	cstn_step2 = relation_constraints(actn_v, args, starts, ends)
	cstn_step3 = variable_constraints(actn_v, args, starts[-1], ends[-1]) #SENSE

	state_step1 = struct_constraints_state()
	state_step2 = relation_constraints_state()
	state_step3 = variable_constraints_state()
	line = 0
	for i in range(len(train_action)):
		print i+1
		input_length = len(train_input[i][0])
		action_step1 = train_action[i][0][1:]
		action_step2 = train_action[i][1]
		action_step3 = train_action[i][2]
		state_step1.reset()

		for act in action_step1:
			#print actn_v.totok(act)
			m = cstn_step1.get_step_mask(state_step1)
			assert m[act] == 1
			cstn_step1.update(act, state_step1)


		state_step2.reset()
		drs_idx = 0
		for k in range(len(action_step1)):
			act_drs = action_step1[k]
			act_next = None
			if k + 1 < len(action_step1):
				act_next = action_step1[k+1]
			if actn_v.totok(act_drs) == "DRS(":
				state_step2.reset_length(input_length)
				state_step2.reset_condition(act_drs, act_next)
				for act in action_step2[drs_idx]:
					if type(act) == types.StringType:
						#print act
						act = int(act[1:-1]) + actn_v.size()
					else:
						#print actn_v.totok(act)
						pass
					m = cstn_step2.get_step_mask(state_step2)
					assert m[act] == 1
					cstn_step2.update(act, state_step2)
				drs_idx += 1

			elif actn_v.totok(act_drs) == "SDRS(":
				state_step2.reset_length(0)
				state_step2.reset_condition(act_drs, act_next)
				for act in action_step2[drs_idx]:
					assert type(act) != types.StringType
					#print actn_v.totok(act)
					m = cstn_step2.get_step_mask(state_step2)
					assert m[act] == 1
					cstn_step2.update(act, state_step2)
				drs_idx += 1

		k_scope = get_k_scope(action_step1, actn_v)
		p_max = get_p_max(action_step1, actn_v)
		b_max = get_b_max(action_step1, actn_v)

		state_step3.reset(p_max, b_max, input_length)
		k = 0
		kk = 0
		sdrs_idx = 0
		for act_drs in action_step1:
			if actn_v.totok(act_drs) in ["DRS(", "SDRS("]:
				if actn_v.totok(act_drs) == "SDRS(":
					state_step3.reset_condition(act_drs, k_scope[sdrs_idx])
					sdrs_idx += 1
				else:
					state_step3.reset_condition(act_drs)
				for act_rel in action_step2[k][:-1]: # rel( rel( )
					if type(act_rel) == types.StringType:
						act_rel = int(act_rel[1:-1]) + actn_v.size()
					state_step3.reset_relation(act_rel)
					for act in action_step3[kk]:
						if type(act) == types.StringType:
							#print act
							act = int(act[1:]) + actn_v.size()
						else:
							#print actn_v.totok(act)
							pass
						#cstn_step3._print_state(state_step3)
						m = cstn_step3.get_step_mask(state_step3)
						assert m[act] == 1
						cstn_step3.update(act, state_step3)
					kk += 1
				k += 1

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
	subparser.add_argument("--train-input", required=True)
	subparser.add_argument("--train-action", required=True)
	subparser.add_argument("--batch-size", type=int, default=1)
	subparser.add_argument("--beam-size", type=int, default=1)
	subparser.add_argument("--check-per-update", type=int, default=1000)
	subparser.add_argument("--eval-per-update", type=int, default=30000)
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
	subparser.add_argument("--beam-size", type=int, default=1)
	subparser.add_argument("--action-dict-path", required=True)
	subparser.add_argument("--use-char", action='store_true')
	subparser.add_argument("--gpu", action='store_true')
	subparser.add_argument("--encoder", default="BILSTM", help="BILSTM, Transformer")
	subparser.add_argument("--const", action="store_true")
	subparser.add_argument("--soft-const", action="store_true")

	subparser = subparsers.add_parser("check")
	subparser.set_defaults(callback=lambda args: run_check(args))
	assign_hypers(subparser, hypers)
	subparser.add_argument("--train-input", default="data/02-21.input")
	subparser.add_argument("--train-action", default="data/02-21.action")
	subparser.add_argument("--action-dict-path", required=True)

	args = parser.parse_args()
	args.callback(args)
