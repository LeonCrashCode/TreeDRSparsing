import torch
import torch.nn as nn
import torch.nn.functional as F

import types
import copy
class Beam:
	def __init__(self):
		self.prev_beam_idx = None
		self.score = None

		self.action_t = None
		self.hidden_t = None
		self.token = None
		self.state = None

		self.output_t = None
		self.next_hidden_t = None

	def show(self):
		print "prev_beam_idx", self.prev_beam_idx
		print "score", self.score
		print "token", self.token
		print "output_t", self.output_t
class decoder(nn.Module):
	def __init__(self, action_size, args, actn_v, constraints):
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

		self.copy_matrix = nn.Linear(self.args.action_hidden_dim, self.args.action_hidden_dim, bias=False)
		self.copy = nn.Linear(self.args.action_hidden_dim, self.args.action_dim)


		self.criterion = nn.NLLLoss()

		self.actn_v = actn_v

		self.cstn1, self.cstn2, self.cstn3 = constraints
	def forward(self, inputs, hidden, encoder_rep_t, train, state, opt):
		if opt == 1:
			return self.forward_1(inputs, hidden, encoder_rep_t, train, state)
		elif opt == 2:
			return self.forward_2(inputs, hidden, encoder_rep_t, train, state)
		elif opt == 3:
			return self.forward_3(inputs, hidden, encoder_rep_t, train, state)
		else:
			assert False, "unrecognized option"
	def forward_1(self, input, hidden, encoder_rep_t, train, state):

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
			hidden_rep = []

			input_t = torch.LongTensor([input])
			if self.args.gpu:
				input_t = input_t.cuda()

			hidden_t = (hidden[0].view(self.args.action_n_layer, 1, -1), hidden[1].view(self.args.action_n_layer, 1, -1))
			action_t = self.embeds(input_t).unsqueeze(1)

			beam = Beam()
			beam.prev_beam_idx = -1
			beam.score = 0

			beam.action_t = action_t
			beam.hidden_t = hidden_t
			beam.state = state

			beamMatrix = [[beam]]
			all_terminal = False
			while not all_terminal:
				#print "=======================", len(beamMatrix), "=================="
				b = 0
				all_terminal = True
				tmp = []
				while b < len(beamMatrix[-1]):
					#pick a beam
					beam = beamMatrix[-1][b]
					#if the beam is terminal, directly extended without prediction 
					#if self.cstn1.isterminal(beam.state):
					#	tmp.append([beam.score, -1, b])
					#	b += 1
					#	continue
					if self.cstn1.isterminal(beam.state):
						tmp.append([beam.score, -1, b])
						b += 1
						continue
					all_terminal = False
					#expand beam
					output, next_hidden = self.lstm(beam.action_t, beam.hidden_t)
					beam.output_t = output
					beam.next_hidden_t = next_hidden

					attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
					attn_weights_t = F.softmax(attn_scores_t, 1)
					attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
					feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
					global_scores_t = self.out(feat_hiddens_t)

					constraint = self.cstn1.get_step_mask(beam.state)
					constraint_t = torch.FloatTensor(constraint).unsqueeze(0)
					if self.args.gpu:
						constraint_t = constraint_t.cuda()
					score_t = global_scores_t + (constraint_t - 1.0) * 1e10

					score_t = F.softmax(score_t, 1)

					scores = score_t.view(-1).data.tolist()
					scores = [ [scores[i], i] for i in range(len(scores))]
					scores.sort(reverse=True)
					for s in scores:
						if constraint[s[-1]] == 0:
							break
						#print s[0],
						average_s = ( s[0] + beam.score * (len(beamMatrix) - 1) ) / len(beamMatrix)
						tmp.append([average_s, s[1], b]) #score tok_idx prev_beam_idx
					b += 1
				#print 
				if all_terminal:
					break

				# candidates
				tmp.sort(reverse=True)
				#print tmp
				b = 0
				beamMatrix.append([])
				while b < len(tmp) and b < self.args.beam_size:
					score, tok_idx, prev_beam_idx = tmp[b]
					#print "===="

					new_beam = Beam()
					new_beam.prev_beam_idx = prev_beam_idx
					new_beam.score = score
					new_beam.token = tok_idx
					new_beam.state = copy.deepcopy(beamMatrix[-2][prev_beam_idx].state)

					if self.cstn1.isterminal(beamMatrix[-2][prev_beam_idx].state):
						new_beam.hidden_t = beamMatrix[-2][prev_beam_idx].hidden_t
					else:
						new_beam.hidden_t = beamMatrix[-2][prev_beam_idx].next_hidden_t # next hidden
						input_t = torch.LongTensor([tok_idx])
						if self.args.gpu:
							input_t = input_t.cuda()
						new_beam.action_t = self.embeds(input_t).view(1, 1, -1)
						self.cstn1.update(tok_idx, new_beam.state)
					#new_beam.show()
					#self.cstn1._print_state(new_beam.state)

					beamMatrix[-1].append(new_beam)
					b += 1
				#if len(beamMatrix) == 5:
				#	exit(1)
			b = 0
			step = len(beamMatrix) - 1
			beam_idx = 0
			hidden = beamMatrix[step][beam_idx].hidden_t
			while True:
				beam = beamMatrix[step][beam_idx]
				if beam.token != -1:
					tokens.append(beam.token)
					hidden_rep.append(beam.output_t)
				step -= 1
				beam_idx = beam.prev_beam_idx
				if step <= 0:
					break
			return tokens[::-1], hidden_rep[::-1], hidden

	def forward_2(self, input, hidden, encoder_rep_t, train, state):
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

			copy_scores_t = torch.bmm(self.copy_matrix(output).transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(output.size(0), -1)
			#copy_scores_t = torch.bmm(torch.bmm(output.transpose(0,1), self.copy_matrix), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(output.size(0), -1)

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
			hidden_rep = []

			action_t = self.struct2rel(input).view(1, 1,-1)

			beam = Beam()
			beam.prev_beam_idx = -1
			beam.score = 0

			beam.action_t = self.struct2rel(input).view(1, 1,-1)
			beam.hidden_t = hidden
			beam.state = state

			beamMatrix = [[beam]]
			all_terminal = False
			while not all_terminal:
				#print "=======================", len(beamMatrix), "=================="
				b = 0
				all_terminal = True
				tmp = []
				while b < len(beamMatrix[-1]):
					#pick a beam
					beam = beamMatrix[-1][b]
					#if the beam is terminal, directly extended without prediction 
					if self.cstn2.isterminal(beam.state):
						tmp.append([beam.score, -1, b])
						b += 1
						continue

					all_terminal = False

					output, next_hidden = self.lstm(beam.action_t, beam.hidden_t)
					beam.output_t = output
					beam.next_hidden_t = next_hidden


					copy_scores_t = torch.bmm(self.copy_matrix(output).transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(output.size(0), -1)
					#copy_scores_t = torch.bmm(torch.bmm(output.transpose(0,1), self.copy_matrix), encoder_rep_t.transpose(0,1).unsqueeze(0)).view(output.size(0), -1)

					attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
					attn_weights_t = F.softmax(attn_scores_t, 1)
					attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
					feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
					global_scores_t = self.out(feat_hiddens_t)

					total_score = torch.cat((global_scores_t, copy_scores_t), 1)

					constraint = self.cstn2.get_step_mask(beam.state)
					constraint_t = torch.FloatTensor(constraint).unsqueeze(0)
					if self.args.gpu:
						constraint_t = constraint_t.cuda()

					total_score = total_score + (constraint_t - 1) * 1e10

					scores = total_score.view(-1).data.tolist()
					scores = [ [scores[i], i] for i in range(len(scores))]
					scores.sort(reverse=True)

					for s in scores:
						if constraint[s[-1]] == 0:
							break
						average_s = ( s[0] + beam.score * (len(beamMatrix) - 1) ) / len(beamMatrix)
						tmp.append([average_s, s[1], b]) #score tok_idx prev_beam_idx
					b += 1
				if all_terminal:
					break

				# candidates
				tmp.sort(reverse=True)
				b = 0
				beamMatrix.append([])
				while b < len(tmp) and b < self.args.beam_size:
					#print "===="
					score, tok_idx, prev_beam_idx = tmp[b]

					new_beam = Beam()
					new_beam.prev_beam_idx = prev_beam_idx
					new_beam.score = score
					new_beam.token = tok_idx
					new_beam.state = copy.deepcopy(beamMatrix[-2][prev_beam_idx].state)

					if self.cstn2.isterminal(beamMatrix[-2][prev_beam_idx].state):
						new_beam.hidden_t = beamMatrix[-2][prev_beam_idx].hidden_t
					else:
						new_beam.hidden_t = beamMatrix[-2][prev_beam_idx].next_hidden_t # next hidden
						if tok_idx >= self.action_size:
							new_beam.action_t = self.copy(encoder_rep_t[tok_idx - self.action_size].view(1, 1, -1))
						else:
							input_t = torch.LongTensor([tok_idx])
							if self.args.gpu:
								input_t = input_t.cuda()
							new_beam.action_t = self.embeds(input_t).view(1, 1, -1)
						self.cstn2.update(tok_idx, new_beam.state)
					#new_beam.show()
					#self.cstn1._print_state(new_beam.state)

					beamMatrix[-1].append(new_beam)
					b += 1
				#if len(beamMatrix) == 2:
				#	exit(1)
			b = 0
			step = len(beamMatrix) - 1
			beam_idx = 0
			hidden = beamMatrix[step][beam_idx].hidden_t

			while True:
				beam = beamMatrix[step][beam_idx]
				if beam.token != -1:
					tokens.append(beam.token)
					hidden_rep.append(beam.output_t)
				step -= 1
				beam_idx = beam.prev_beam_idx
				if step <= 0:
					break
			return tokens[::-1], hidden_rep[::-1], hidden

	def forward_3(self, input, hidden, encoder_rep_t, train, state):
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
			#hidden_rep = []

			action_t = self.rel2var(input).view(1, 1,-1)

			beam = Beam()
			beam.prev_beam_idx = -1
			beam.score = 0

			beam.action_t = action_t
			beam.hidden_t = hidden
			beam.state = state

			beamMatrix = [[beam]]
			all_terminal = False
			while not all_terminal:
				#print "=======================", len(beamMatrix), "=================="
				b = 0
				all_terminal = True
				tmp = []
				while b < len(beamMatrix[-1]):
					#pick a beam
					beam = beamMatrix[-1][b]
					#if the beam is terminal, directly extended without prediction 
					if self.cstn3.isterminal(beam.state):
						tmp.append([beam.score, -1, b])
						b += 1
						continue

					all_terminal = False

					#expand beam
					output, next_hidden = self.lstm(beam.action_t, beam.hidden_t)
					beam.output_t = output
					beam.next_hidden_t = next_hidden

					attn_scores_t = torch.bmm(output.transpose(0,1), encoder_rep_t.transpose(0,1).unsqueeze(0))[0]
					attn_weights_t = F.softmax(attn_scores_t, 1)
					attn_hiddens_t = torch.bmm(attn_weights_t.unsqueeze(0),encoder_rep_t.unsqueeze(0))[0]
					feat_hiddens_t = self.feat_tanh(self.feat(torch.cat((attn_hiddens_t, action_t.view(output.size(0),-1)), 1)))
					global_scores_t = self.out(feat_hiddens_t)

					constraint = self.cstn3.get_step_mask(beam.state)
					constraint_t = torch.FloatTensor(constraint).unsqueeze(0)
					if self.args.gpu:
						constraint_t = constraint_t.cuda()
					score_t = global_scores_t + (constraint_t - 1.0) * 1e10

					scores = score_t.view(-1).data.tolist()
					scores = [ [scores[i], i] for i in range(len(scores))]
					scores.sort(reverse=True)

					for s in scores:
						if constraint[s[-1]] == 0:
							break
						average_s = ( s[0] + beam.score * (len(beamMatrix) - 1) ) / len(beamMatrix)
						tmp.append([average_s, s[1], b]) #score tok_idx prev_beam_idx
					b += 1
				if all_terminal:
					break

				# candidates
				tmp.sort(reverse=True)
				b = 0
				beamMatrix.append([])
				while b < len(tmp) and b < self.args.beam_size:
					#print "===="
					score, tok_idx, prev_beam_idx = tmp[b]
					
					new_beam = Beam()
					new_beam.prev_beam_idx = prev_beam_idx
					new_beam.score = score
					new_beam.token = tok_idx
					new_beam.state = copy.deepcopy(beamMatrix[-2][prev_beam_idx].state)

					if self.cstn3.isterminal(beamMatrix[-2][prev_beam_idx].state):
						new_beam.hidden_t = beamMatrix[-2][prev_beam_idx].hidden_t
					else:
						new_beam.hidden_t = beamMatrix[-2][prev_beam_idx].next_hidden_t # next hidden
						input_t = torch.LongTensor([tok_idx])
						if self.args.gpu:
							input_t = input_t.cuda()
						new_beam.action_t = self.embeds(input_t).view(1, 1, -1)
						self.cstn3.update(tok_idx, new_beam.state)

					#new_beam.show()
					#self.cstn3.__print_state(new_beam.state)

					beamMatrix[-1].append(new_beam)
					b += 1

			b = 0
			step = len(beamMatrix) - 1
			beam_idx = 0
			hidden = beamMatrix[step][beam_idx].hidden_t
			while True:
				beam = beamMatrix[step][beam_idx]
				if beam.token != -1:
					tokens.append(beam.token)
					#hidden_rep.apend(beam.output_t)
				step -= 1
				beam_idx = beam.prev_beam_idx
				if step <= 0:
					break
			return tokens[::-1], None, hidden

