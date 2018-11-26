import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)

    def forward(self, input_t, comb, sep, train=True):
        hidden_t = self.inithidden()
        if train:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0
        output_t, hidden_t = self.lstm(input_t.unsqueeze(1), hidden_t)


        copy_rep = []
        for i in range(len(comb)):
            copy_rep.append([])
            for idx in comb[i]:
                copy_rep[-1].append(output_t[idx])
            copy_rep[-1] = (torch.sum(torch.cat(copy_rep[-1]),0)/(len(comb[i]))).unsqueeze(0)
        copy_rep = torch.cat(copy_rep, 0)

        sent_rep = []
        for i in range(len(sep)-1):
            s = sep[i]
            e = sep[i+1]
            sent_rep.append(torch.cat([output_t[e].view(2,-1)[0], output_t[s].view(2,-1)[1]]).view(1, -1))
        sent_rep = torch.cat(sent_rep, 0).unsqueeze(0)
        #output_t is 1 x n x H, where n = n0 + n1
        #copy_rep is [1 x ni x H], m list
        #sent_rep is 1 x m x H
        return output_t.transpose(0,1), sent_rep, copy_rep, hidden_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
            return result

class encoder_srnn(nn.Module):
    def __init__(self, args):
        super(encoder_srnn, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)
        self.sent_lstm = nn.LSTM(args.bilstm_hidden_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)
        self.word2sent = nn.Linear(args.bilstm_hidden_dim*2, args.bilstm_hidden_dim)
    def forward(self, input_t, combs, seps, train=True):
        hidden_t = self.inithidden()
        if train:
            self.lstm.dropout = self.args.dropout_f
            self.sent_lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0
            self.sent_lstm.dropout = 0

        output_t, hidden_t = self.lstm(input_t.unsqueeze(1), hidden_t)

        assert len(seps) -1 == len(combs)

        copy_rep_s = []
        for i in range(len(seps)-1):
            s = seps[i]
            e = seps[i+1]
            sent = output_t[s+1:e]
            comb = combs[i]

            copy_rep = []
            for j in range(len(comb)):
                copy_rep.append([])
                for idx in comb[j]:
                    copy_rep[-1].append(sent[idx])
                copy_rep[-1] = (torch.sum(torch.cat(copy_rep[-1]),0)/(len(comb[j]))).unsqueeze(0)
            copy_rep = torch.cat(copy_rep, 0)
            copy_rep_s.append(copy_rep)

        sent_rep = []
        for i in range(len(seps)-1):
            s = seps[i]
            e = seps[i+1]
            sent_rep.append(torch.cat([output_t[e].view(2,-1)[0], output_t[s].view(2,-1)[1]]).view(1, -1))
        sent_rep = torch.cat(sent_rep, 0).unsqueeze(1)

        shidden_t = self.inithidden()
        sent_rep, _ = self.sent_lstm(self.word2sent(sent_rep), shidden_t)
        #output_t is 1 x n x H, where n = n0 + n1
        #copy_rep is [1 x ni x H], m list
        #sent_rep is 1 x m x H
        return output_t.transpose(0,1), sent_rep.transpose(0,1), copy_rep_s, hidden_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
            return result