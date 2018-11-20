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
        self.sent2input = nn.Linear(args.bilstm_hidden_dim * 2, args.bilstm_hidden_dim, bias=False)
        self.doc_lstm = nn.LSTM(args.bilstm_hidden_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True)
    def forward(self, input_t, comb, train=True):
        hidden_t = self.inithidden()
        if train:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0

        output_t = []
        sent_rep = []
        for one_input_t in input_t:
            hidden_t = self.inithidden()
            one_output_t, _ = self.lstm(one_input_t.unsqueeze(1), hidden_t)
            output_t.append(one_output_t)
            sent_rep.append(torch.sum(one_output_t,0).unsqueeze(0) / one_output_t.size(0)) #[1 x 1 x H]
        
        #output_t is [ni x 1 x H]

        encoder_rep = []
        for i in range(len(comb)):
            encoder_rep.append([])
            for r,c in comb[i]:
                encoder_rep[-1].append(output_t[r][c])
            encoder_rep[-1] = (torch.sum(torch.cat(encoder_rep[-1]),0)/(len(comb[i]))).unsqueeze(0)
        encoder_rep = torch.cat(encoder_rep, 0) 

        sent_input = self.sent2input(torch.cat(sent_rep, 0)) #[m x 1 x H]

        hidden_t = self.inithidden()
        sent_rep, hidden_t = self.doc_lstm(sent_input, hidden_t)

        # sent_rep is m x 1 x H
        output_t = [x.transpose(0,1) for x in output_t] # [1 x ni x H]

        return output_t, sent_rep.transpose(0,1), encoder_rep, hidden_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, 1, self.args.bilstm_hidden_dim, requires_grad=True))
            return result