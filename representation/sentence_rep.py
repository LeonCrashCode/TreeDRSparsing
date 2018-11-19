from random import random

import torch
import torch.nn as nn

class sentence_rep(nn.Module):
	def __init__(self, word_size, char_size, pretrain, extra_vl_size, args):
		super(sentence_rep,self).__init__()
		self.args = args
		self.word_embeds = nn.Embedding(word_size, args.word_dim, padding_idx=0)
		info_dim = args.word_dim
		if args.use_char:
			assert False, "no implementation"
			self.char_embeds = nn.Embedding(char_size, args.char_dim, padding_idx=0)
			self.lstm = nn.LSTM(args.char_dim, args.char_hidden_dim, num_layers=args.char_n_layer, bidirectional=True)
			info_dim += args.char_hidden_dim*2*3
		if args.pretrain_path:
			self.pretrain_embeds = nn.Embedding(pretrain.size(), args.pretrain_dim, padding_idx=0)
			self.pretrain_embeds.weight = nn.Parameter(torch.FloatTensor(pretrain.vectors()), False)
			info_dim += args.pretrain_dim
		if args.extra_dim_list:
			dims = args.extra_dim_list.split(",")
			"""
			self.extra_embeds = []
			for i, size in enumerate(extra_vl_size):
				if args.gpu:
					self.extra_embeds.append(nn.Embedding(size, int(dims[i])).cuda())
				else:
					self.extra_embeds.append(nn.Embedding(size, int(dims[i])))
				info_dim += int(dims[i])
			"""
			assert len(dims) == len(extra_vl_size)
			assert len(dims) <= 5, "5 extra embeds at most"
			if len(dims) >= 1:
				self.extra_embeds1 = nn.Embedding(extra_vl_size[0], int(dims[0]), padding_idx=0)
				info_dim += int(dims[0])
			if len(dims) >= 2:
				self.extra_embeds2 = nn.Embedding(extra_vl_size[1], int(dims[1]), padding_idx=0)
				info_dim += int(dims[1])
			if len(dims) >= 3:
				self.extra_embeds3 = nn.Embedding(extra_vl_size[2], int(dims[2]), padding_idx=0)
				info_dim += int(dims[2])
			if len(dims) >= 4:
				self.extra_embeds4 = nn.Embedding(extra_vl_size[3], int(dims[3]), padding_idx=0)
				info_dim += int(dims[3])
			if len(dims) >= 5:
				self.extra_embeds5 = nn.Embedding(extra_vl_size[4], int(dims[4]), padding_idx=0)
				info_dim += int(dims[4])

		self.info2input = nn.Linear(info_dim, args.input_dim)
		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(self.args.dropout_f)

	def forward(self, batch_instance, singleton_idx_dict=None, train=True):

		word_sequence = [[]]
		for i, instance in enumerate(batch_instance, 0):
			for j, widx in enumerate(instance[0], 0):
				if train and (widx in singleton_idx_dict) and random() < self.args.single_f:
					word_sequence[-1].append(instance[3][i])
				else:
					word_sequence[-1].append(widx)
			word_sequence.append([])
		word_sequence.pop() # pop last empty list
		#print [len(x) for x in word_sequence]
		word_sequence = [torch.LongTensor(x) for x in word_sequence]
		if self.args.gpu:
			word_sequence = [x.cuda() for x in word_sequence]
		word_order, sorted_by_length = zip(*sorted(enumerate(word_sequence), key = lambda x: len(x[1]), reverse=True))
		lengths = [len(i) for i in sorted_by_length]
		#print word_order
		padded_word_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)

		padded_word_sequence_embeddings = self.word_embeds(padded_word_sequence)

		if train:
			padded_word_sequence_embeddings = self.dropout(padded_word_sequence_embeddings)
		#print word_t, word_t.size()
		if self.args.use_char:
			assert False, "no implementation"
			char_ts = []
			for char_instance in instance[1]:
				char_t = torch.LongTensor(char_instance)
				if self.args.gpu:
					char_t = char_t.cuda()
				char_t = self.char_embeds(char_t)
				char_hidden_t = self.initcharhidden()
				char_t, _ = self.lstm(char_t.unsqueeze(1), char_hidden_t)
				char_t_avg = torch.sum(char_t,0) / char_t.size(0)
				char_t_max = torch.max(char_t,0)[0]
				char_t_min = torch.min(char_t,0)[0]
				char_t_per_word = torch.cat((char_t_avg, char_t_max, char_t_min), 1)
				if train:
					char_t_per_word = self.dropout(char_t_per_word)
				char_ts.append(char_t_per_word)
			char_t = torch.cat(char_ts, 0)
			word_t = torch.cat((word_t, char_t), 1)
		#print word_t, word_t.size()
		if self.args.pretrain_path:
			pretrain_sequence = [ instance[2] for instance in batch_instance]
			pretrain_sequence = [torch.LongTensor(x) for x in pretrain_sequence]
			if self.args.gpu:
				pretrain_sequence = [x.cuda() for x in pretrain_sequence]
			pretrain_order, sorted_by_length = zip(*sorted(enumerate(pretrain_sequence), key = lambda x: len(x[1]), reverse=True))
			assert word_order == pretrain_order 
			padded_pretrain_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)
			padded_pretrain_sequence_embeddings = self.pretrain_embeds(padded_pretrain_sequence)
			padded_word_sequence_embeddings = torch.cat((padded_word_sequence_embeddings, padded_pretrain_sequence_embeddings), 2)
		#print word_t, word_t.size()
		if self.args.extra_dim_list:
			"""
			for i, extra_embeds in enumerate(self.extra_embeds):
				extra_t = torch.LongTensor(instance[4+i])
				if self.args.gpu:
					extra_t = extra_t.cuda()
				extra_t = extra_embeds(extra_t)
				if not test:
					extra_t = self.dropout(extra_t)
				word_t = torch.cat((word_t, extra_t), 1)
			"""
			if len(instance)-4 >= 1:
				extra_sequence = [ instance[4+0] for instance in batch_instance]
				extra_sequence = [torch.LongTensor(x) for x in extra_sequence]
				if self.args.gpu:
					extra_sequence = [x.cuda() for x in extra_sequence]
				extra_order, sorted_by_length = zip(*sorted(enumerate(extra_sequence), key = lambda x: len(x[1]), reverse=True))
				assert word_order == extra_order
				padded_extra_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)
				padded_extra_sequence_embeddings = self.extra_embeds1(padded_extra_sequence)
				padded_word_sequence_embeddings = torch.cat((padded_word_sequence_embeddings, padded_extra_sequence_embeddings), 2)
			if len(instance)-4 >= 2:
				extra_sequence = [ instance[4+1] for instance in batch_instance]
				extra_sequence = [torch.LongTensor(x) for x in extra_sequence]
				if self.args.gpu:
					extra_sequence = [x.cuda() for x in extra_sequence]
				extra_order, sorted_by_length = zip(*sorted(enumerate(extra_sequence), key = lambda x: len(x[1]), reverse=True))
				assert word_order == extra_order
				padded_extra_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)
				padded_extra_sequence_embeddings = self.extra_embeds2(padded_extra_sequence)
				padded_word_sequence_embeddings = torch.cat((padded_word_sequence_embeddings, padded_extra_sequence_embeddings), 2)
			if len(instance)-4 >= 3:
				extra_sequence = [ instance[4+2] for instance in batch_instance]
				extra_sequence = [torch.LongTensor(x) for x in extra_sequence]
				if self.args.gpu:
					extra_sequence = [x.cuda() for x in extra_sequence]
				extra_order, sorted_by_length = zip(*sorted(enumerate(extra_sequence), key = lambda x: len(x[1]), reverse=True))
				assert word_order == extra_order
				padded_extra_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)
				padded_extra_sequence_embeddings = self.extra_embeds3(padded_extra_sequence)
				padded_word_sequence_embeddings = torch.cat((padded_word_sequence_embeddings, padded_extra_sequence_embeddings), 2)
			if len(instance)-4 >= 4:
				extra_sequence = [ instance[4+3] for instance in batch_instance]
				extra_sequence = [torch.LongTensor(x) for x in extra_sequence]
				if self.args.gpu:
					extra_sequence = [x.cuda() for x in extra_sequence]
				extra_order, sorted_by_length = zip(*sorted(enumerate(extra_sequence), key = lambda x: len(x[1]), reverse=True))
				assert word_order == extra_order
				padded_extra_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)
				padded_extra_sequence_embeddings = self.extra_embeds4(padded_extra_sequence)
				padded_word_sequence_embeddings = torch.cat((padded_word_sequence_embeddings, padded_extra_sequence_embeddings), 2)
			if len(instance)-4 >= 5:
				extra_sequence = [ instance[4+4] for instance in batch_instance]
				extra_sequence = [torch.LongTensor(x) for x in extra_sequence]
				if self.args.gpu:
					extra_sequence = [x.cuda() for x in extra_sequence]
				extra_order, sorted_by_length = zip(*sorted(enumerate(extra_sequence), key = lambda x: len(x[1]), reverse=True))
				assert word_order == extra_order
				padded_extra_sequence =  nn.utils.rnn.pad_sequence(sorted_by_length, batch_first=True, padding_value=0)
				padded_extra_sequence_embeddings = self.extra_embeds5(padded_extra_sequence)
				padded_word_sequence_embeddings = torch.cat((padded_word_sequence_embeddings, padded_extra_sequence_embeddings), 2)

		#print word_embeddings, word_embeddings.size()
		padded_word_sequence_embeddings = self.tanh(self.info2input(padded_word_sequence_embeddings))
		#print word_embeddings, word_embeddings.size()

		packed_padded_word_sequence_embeddings = nn.utils.rnn.pack_padded_sequence(padded_word_sequence_embeddings, lengths, batch_first=True)
		return packed_padded_word_sequence_embeddings, word_order

	def initcharhidden(self):
		if self.args.gpu:
			result = (torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).cuda(),
				torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).cuda())
			return result
		else:
			result = (torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True),
				torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True))
			return result
