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
        self.lstm = nn.LSTM(args.input_dim, args.bilstm_hidden_dim, num_layers=args.bilstm_n_layer, bidirectional=True, batch_first=True)

    def forward(self, input_t, batch_comb, order, train=True):
        hidden_t = self.inithidden()
        if train:
            self.lstm.dropout = self.args.dropout_f
        else:
            self.lstm.dropout = 0

        output_t, hidden_t = self.lstm(input_t, hidden_t)

        padded_sequence, length = nn.utils.rnn.pad_packed_sequence(output_t, batch_first=True, padding_value=0.0)


        batch_encoder_rep = []
        for idx in range(padded_sequence.size(0)):
            o = order[idx]
            comb = batch_comb[o]
            encoder_rep = []
            for i in range(padded_sequence.size(1)):
                if i >= len(comb):
                    encoder_rep.append(padded_sequence[idx][i].view(1,-1))
                    continue
                encoder_rep.append([padded_sequence[idx][i].view(1,-1)])
                for j in comb[i]:
                    encoder_rep[-1].append(padded_sequence[idx][j].view(1,-1))
                encoder_rep[-1] = (torch.sum(torch.cat(encoder_rep[-1],0), 0) / (len(comb[i]) + 1)).unsqueeze(0)
            encoder_rep = torch.cat(encoder_rep[1:-1], 0)
            batch_encoder_rep.append(encoder_rep.unsqueeze(0))

        reorder_batch_encoder_rep = [None for x in range(len(batch_encoder_rep))]

        for i, rep in enumerate(batch_encoder_rep):
            reorder_batch_encoder_rep[order[i]] = rep

        return torch.cat(reorder_batch_encoder_rep,0), hidden_t

    def inithidden(self):
        if self.args.gpu:
            result = (torch.zeros(2*self.args.bilstm_n_layer, self.args.batch_size, self.args.bilstm_hidden_dim, requires_grad=True).cuda(),
                torch.zeros(2*self.args.bilstm_n_layer, self.args.batch_size, self.args.bilstm_hidden_dim, requires_grad=True).cuda())
            return result
        else:
            result = (torch.zeros(2*self.args.bilstm_n_layer, self.args.batch_size, self.args.bilstm_hidden_dim, requires_grad=True),
                torch.zeros(2*self.args.bilstm_n_layer, self.args.batch_size, self.args.bilstm_hidden_dim, requires_grad=True))
            return result