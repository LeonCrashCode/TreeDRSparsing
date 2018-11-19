from random import random

import torch
import torch.nn as nn

class sentence_rep(nn.Module):
	def __init__(self, word_size, char_size, pretrain, extra_vl_size, args):
		super(sentence_rep,self).__init__()
		self.args = args
		self.word_embeds = nn.Embedding(word_size, args.word_dim)
		info_dim = args.word_dim
		if args.use_char:
			self.char_embeds = nn.Embedding(char_size, args.char_dim)
			self.lstm = nn.LSTM(args.char_dim, args.char_hidden_dim, num_layers=args.char_n_layer, bidirectional=True)
			info_dim += args.char_hidden_dim*2*3
		if args.pretrain_path:
			self.pretrain_embeds = nn.Embedding(pretrain.size(), args.pretrain_dim)
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
				self.extra_embeds1 = nn.Embedding(extra_vl_size[0], int(dims[0]))
				info_dim += int(dims[0])
			if len(dims) >= 2:
				self.extra_embeds2 = nn.Embedding(extra_vl_size[1], int(dims[1]))
				info_dim += int(dims[1])
			if len(dims) >= 3:
				self.extra_embeds3 = nn.Embedding(extra_vl_size[2], int(dims[2]))
				info_dim += int(dims[2])
			if len(dims) >= 4:
				self.extra_embeds4 = nn.Embedding(extra_vl_size[3], int(dims[3]))
				info_dim += int(dims[3])
			if len(dims) >= 5:
				self.extra_embeds5 = nn.Embedding(extra_vl_size[4], int(dims[4]))
				info_dim += int(dims[4])

		self.info2input = nn.Linear(info_dim, args.input_dim)
		self.tanh = nn.Tanh()
		self.dropout = nn.Dropout(self.args.dropout_f)

	def forward(self, instances, singleton_idx_dict=None, train=True):
		reps = []
		for instance in instances:
			word_sequence = []
			for i, widx in enumerate(instance[0], 0):
				if train and (widx in singleton_idx_dict) and random() < self.args.single_f:
				#if False:
					word_sequence.append(instance[3][i])
				else:
					word_sequence.append(widx)

			word_t = torch.LongTensor(word_sequence)
			if self.args.gpu:
				word_t = word_t.cuda()
			word_t = self.word_embeds(word_t)

			if train:
				word_t = self.dropout(word_t)
			#print word_t, word_t.size()
			if self.args.use_char:
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
				pretrain_t = torch.LongTensor(instance[2])
				if self.args.gpu:
					pretrain_t = pretrain_t.cuda()
				pretrain_t = self.pretrain_embeds(pretrain_t)
				word_t = torch.cat((word_t, pretrain_t), 1)
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
					extra_t = torch.LongTensor(instance[4+0])
					if self.args.gpu:
						extra_t = extra_t.cuda()
					extra_t = self.extra_embeds1(extra_t)
					word_t = torch.cat((word_t, extra_t), 1)
				if len(instance)-4 >= 2:
					extra_t = torch.LongTensor(instance[4+1])
					if self.args.gpu:
						extra_t = extra_t.cuda()
					extra_t = self.extra_embeds1(extra_t)
					word_t = torch.cat((word_t, extra_t), 1)
				if len(instance)-4 >= 3:
					extra_t = torch.LongTensor(instance[4+2])
					if self.args.gpu:
						extra_t = extra_t.cuda()
					extra_t = self.extra_embeds1(extra_t)
					word_t = torch.cat((word_t, extra_t), 1)
				if len(instance)-4 >= 4:
					extra_t = torch.LongTensor(instance[4+3])
					if self.args.gpu:
						extra_t = extra_t.cuda()
					extra_t = self.extra_embeds1(extra_t)
					word_t = torch.cat((word_t, extra_t), 1)
				if len(instance)-4 >= 5:
					extra_t = torch.LongTensor(instance[4+4])
					if self.args.gpu:
						extra_t = extra_t.cuda()
					extra_t = self.extra_embeds1(extra_t)
					word_t = torch.cat((word_t, extra_t), 1)

			#print word_embeddings, word_embeddings.size()
			word_t = self.tanh(self.info2input(word_t))
			reps.append(word_t)
			#print word_embeddings, word_embeddings.size()
		return reps

	def initcharhidden(self):
		if self.args.gpu:
			result = (torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).cuda(),
				torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True).cuda())
			return result
		else:
			result = (torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True),
				torch.zeros(2*self.args.char_n_layer, 1, self.args.char_hidden_dim, requires_grad=True))
			return result
