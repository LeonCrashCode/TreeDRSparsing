import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, args):
        super(encoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)

    def forward(self, input_t, train=True):
        hidden_t = self.inithidden()
        if not test:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0
        output_t, _ = self.lstm(input_t.unsqueeze(1), hidden_t)
        return output_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
            return result

class comb_encoder(nn.Module):
    def __init__(self, args):
        super(comb_encoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)

    def forward(self, input_t, combs, sep, train=True):
        hidden_t = self.inithidden()
        if train:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0
        output_t, hidden_t = self.lstm(input_t.unsqueeze(1), hidden_t)

        copy_rep = []
        for comb in combs:
            encoder_rep = []
            for i in range(len(comb)):
                encoder_rep.append([])
                for idx in comb[i]:
                    encoder_rep[-1].append(output_t[idx])
                encoder_rep[-1] = (torch.sum(torch.cat(encoder_rep[-1]),0)/(len(comb[i]))).unsqueeze(0)
            encoder_rep = torch.cat(encoder_rep, 0)
            copy_rep.append(encoder_rep.unsqueeze(0))

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