import torch
import torch.nn as nn
import torch.nn.functional as F

import types

class decoder(nn.Module):
	def __init__(self, action_size, args, actn_v):
		super(decoder, self).__init__()
		self.action_size = action_size
		self.args = args

		self.dropout = nn.Dropout(self.args.dropout_f)
		self.embeds = nn.Embedding(self.action_size, self.args.action_dim)

		self.struct2rel = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)
		self.rel2var = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)

		self.lstm = nn.LSTM(self.args.action_dim, self.args.action_hidden_dim, num_layers= self.args.action_n_layer)

		self.feat = nn.Linear(self.args.action_hidden_dim + self.args.action_dim, self.args.action_feature_dim)
		self.feat_tanh = nn.Tanh()

		self.out = nn.Linear(self.args.action_feature_dim, self.action_size)

		self.copy_matrix = torch.randn(1, self.args.action_hidden_dim, self.args.action_hidden_dim)
		self.copy = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)
		if self.args.gpu:
			self.copy_matrix = self.copy_matrix.cuda()

		self.criterion = nn.NLLLoss()

		self.actn_v = actn_v
	def forward(self, inputs, hidden, encoder_rep_t, train, constraints, opt):
		if opt == 1:
			return self.forward_1(inputs, hidden, encoder_rep_t, train, constraints)
		elif opt == 2:
			return self.forward_2(inputs, hidden, encoder_rep_t, train, constraints)
		elif opt == 3:
			return self.forward_3(inputs, hidden, encoder_rep_t, train, constraints)
		else:
			assert False, "unrecognized option"
	def forward_1(self, input, hidden, encoder_rep_t, train, constraints):

		if train:
			self.lstm.dropout = self.args.dropout_f
			input_t = torch.LongTensor(input[:-1])
			if self.args.gpu:
				input_t = input_t.cuda()
			action_t = self.embeds(input_t).unsqueeze(1)
			action_t = self.dropout(action_t)

			output, hidden = self.lstm(action_t, hidden)

			attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
			attn_weights_t = F.softmax(attn_scores_t, 1)
			attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
			feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
			global_scores_t = self.out(feat_hiddens_t)

			log_softmax_output_t = F.log_softmax(global_scores_t, 1)

			action_g_t = torch.LongTensor(input[1:])
			if self.args.gpu:
				action_g_t = action_g_t.cuda()

			loss_t = self.criterion(log_softmax_output_t, action_g_t)

			return loss_t, output, hidden
		else:
			self.lstm.dropout = 0.0
			tokens = []
			input_t = torch.LongTensor([input])
			if self.args.gpu:
				input_t = input_t.cuda()
			hidden_rep = []
			hidden = (hidden[0].view(self.args.action_n_layer, 1, -1), hidden[1].view(self.args.action_n_layer, 1, -1))
			action_t = self.embeds(input_t).unsqueeze(1)
			while True:
				constraint = constraints.get_step_mask()
				constraint_t = torch.FloatTensor(constraint).unsqueeze(0)
				if self.args.gpu:
					constraint_t = constraint_t.cuda()

				output, hidden = self.lstm(action_t, hidden)
				hidden_rep.append(output)

				attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
				attn_weights_t = F.softmax(attn_scores_t, 1)
				attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
				feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
				global_scores_t = self.out(feat_hiddens_t)

				score = global_scores_t + (constraint_t - 1.0) * 1e10

				_, input_t = torch.max(score,1)
				idx = input_t.view(-1).data.tolist()[0]
				tokens.append(idx)

				constraints.update(idx)

				if constraints.isterminal():
					break

				action_t = self.embeds(input_t).view(1, 1, -1)
			return tokens, hidden_rep, hidden

	def forward_2(self, input, hidden, encoder_rep_t, train, constraints):
		if train:
			self.lstm.dropout = self.args.dropout_f
			List = []
			g_List = []
			for struct, rels in input:
				List.append(self.struct2rel(struct).view(1, 1, -1))
				g_List += rels
				for rel in rels[:-1]: # rel( rel( rel( )
					assert type(rel) != types.NoneType
					if type(rel) == types.StringType:
						List.append(self.copy(encoder_rep_t[int(rel[1:-1])].view(1, 1, -1)))
						#List.append(self.copy(input_rep_t[int(rel[1:-1])+1].view(1, 1, -1)))
					else:
						rel_t = torch.LongTensor([rel])
						if self.args.gpu:
							rel_t = rel_t.cuda()
						List.append(self.embeds(rel_t).unsqueeze(0))

			action_t = torch.cat(List, 0)
			action_t = self.dropout(action_t)

			output, hidden = self.lstm(action_t, hidden)

			copy_scores_t = torch.bmm(torch.bmm(output.transpose(0,1), self.copy_matrix), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(output.size(0), -1)

			attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
			attn_weights_t = F.softmax(attn_scores_t, 1)
			attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
			feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
			global_scores_t = self.out(feat_hiddens_t)

			#print global_scores_t.size()
			#print copy_scores_t.size()
			total_score = torch.cat((global_scores_t, copy_scores_t), 1)
			#print total_score.size()
			log_softmax_output_t = F.log_softmax(total_score, 1)

			for i in range(len(g_List)):
		   		if type(g_List[i]) == types.StringType:
		   			g_List[i] = int(g_List[i][1:-1]) + self.action_size

			action_g_t = torch.LongTensor(g_List)
			if self.args.gpu:
				action_g_t = action_g_t.cuda()

			#for x in g_List:
			#	if x >= self.action_size:
			#		print "$"+str(x-self.action_size)+"(",
			#	else:
			#		print self.actn_v.totok(x),
			#print
			#
			#exit(1)

			loss_t = self.criterion(log_softmax_output_t, action_g_t)

			return loss_t, output, hidden

		else:
			self.lstm.dropout = 0.0
			tokens = []
			hidden_reps = []
			action_t = self.struct2rel(input).view(1, 1,-1)

			while True:
				constraint = constraints.get_step_mask()
				constraint_t = torch.FloatTensor(constraint).unsqueeze(0)
				if self.args.gpu:
					constraint_t = constraint_t.cuda()

				output, hidden = self.lstm(action_t, hidden)
				hidden_reps.append(output)

				copy_scores_t = torch.bmm(torch.bmm(output.transpose(0,1), self.copy_matrix), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(output.size(0), -1)

				attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
				attn_weights_t = F.softmax(attn_scores_t, 1)
				attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
				feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
				global_scores_t = self.out(feat_hiddens_t)

				total_score = torch.cat((global_scores_t, copy_scores_t), 1)

				total_score = total_score + (constraint_t - 1) * 1e10

				_, input_t = torch.max(total_score,1)

				idx = input_t.view(-1).data.tolist()[0]
				tokens.append(idx)
				constraints.update(idx)
				if constraints.isterminal():
					break

				if idx >= self.action_size:
					action_t = self.copy(encoder_rep_t[idx - self.action_size].view(1, 1, -1))
				else:
					action_t = self.embeds(input_t).view(1, 1, -1)

			return tokens, hidden_reps, hidden

	def forward_3(self, input, hidden, encoder_rep_t, train, constraints):
		if train:
			self.lstm.dropout = self.args.dropout_f
			List = []
			g_List = []
			for rel, var in input:
				List.append(self.rel2var(rel).view(1, 1, -1))
				g_List += var
				var_t = torch.LongTensor(var[:-1])
				if self.args.gpu:
					var_t = var_t.cuda()
				List.append(self.embeds(var_t).unsqueeze(1))

			#print [x.size() for x in List]
			action_t = torch.cat(List, 0)
			#print action_t.size()
			action_t = self.dropout(action_t)

			output, hidden = self.lstm(action_t, hidden)

			attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
			attn_weights_t = F.softmax(attn_scores_t, 1)
			attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
			feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
			global_scores_t = self.out(feat_hiddens_t)


			log_softmax_output_t = F.log_softmax(global_scores_t, 1)

			action_g_t = torch.LongTensor(g_List)
			if self.args.gpu:
				action_g_t = action_g_t.cuda()
			loss_t = self.criterion(log_softmax_output_t, action_g_t)

			return loss_t, output, hidden

		else:
			self.lstm.dropout = 0.0
			tokens = []
			action_t = self.rel2var(input).view(1, 1,-1)
			while True:
				constraint = constraints.get_step_mask()
				constraint_t = torch.FloatTensor(constraint).unsqueeze(0)
				if self.args.gpu:
					constraint_t = constraint_t.cuda()

				output, hidden = self.lstm(action_t, hidden)

				attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
				attn_weights_t = F.softmax(attn_scores_t, 1)
				attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
				feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
				global_scores_t = self.out(feat_hiddens_t)

				score = global_scores_t + (constraint_t - 1) * 1e10

				_, input_t = torch.max(score,1)
				idx = input_t.view(-1).data.tolist()[0]
				tokens.append(idx)

				constraints.update(idx)

				if constraints.isterminal():
					break
				action_t = self.embeds(input_t).view(1, 1, -1)
				
			return tokens, None, hidden

