import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class GRUUint(nn.Module):

    def __init__(self, hid_dim, act):
        super(GRUUint, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim)
        self.lin_z1 = nn.Linear(hid_dim, hid_dim)
        self.lin_r0 = nn.Linear(hid_dim, hid_dim)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim)
        self.lin_h0 = nn.Linear(hid_dim, hid_dim)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        z = (self.lin_z0(a) + self.lin_z1(x)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r)))
        return h * z + x * (1 - z)


class GraphLayer(gnn.MessagePassing):

    def __init__(self, in_dim, out_dim, dropout,
                 act=torch.relu, bias=False, step=2):
        super(GraphLayer, self).__init__(aggr='add')
        self.step = step
        self.act = act
        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.gru = GRUUint(out_dim, act=act)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
        print(self.step)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, g):
        x = self.encode(x)
        x = self.act(x)
        for _ in range(self.step):
            a = self.propagate(edge_index=g.edge_index, x=x, edge_attr=self.dropout(g.edge_attr))
            x = self.gru(x, a)
            # x = x * tfidf
        x = self.graph2batch(x, g.length)
        return x

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.unsqueeze(-1)

    def update(self, inputs):
        return inputs

    def graph2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout,
                 act=torch.relu, bias=False):
        super(ReadoutLayer, self).__init__()
        self.act = act
        self.bias = bias
        self.att = nn.Linear(in_dim, 1, bias=True)
        self.emb = nn.Linear(in_dim, in_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        att = self.att(x).sigmoid()
        emb = self.act(self.emb(x))
        x = att * emb
        x = self.__max(x, mask) + self.__mean(x, mask)
        x = self.mlp(x)
        return x

    def __max(self, x, mask):
        return (x + (mask - 1) * 1e9).max(1)[0]

    def __mean(self, x, mask):
        return (x * mask).sum(1) / mask.sum(1)


class TextBILSTM(nn.Module):

    def __init__(self,embeding,num_classes,keep_dropout,hidden_dims,
                 rnn_layers,ettention=True):
        super(TextBILSTM, self).__init__()
        self.embeding = embeding
        self.num_classes=num_classes
        self.keep_dropout = keep_dropout
        self.attention = ettention
        #self.l2_reg_lambda = self.l2_reg_lambda
        self.hidden_dims = hidden_dims
        self.rnn_layers=rnn_layers
        self.build_model()
        self.reset_parameters()

    def build_model(self):

        # attention layer
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True)
        )
        # self.attention_weights = self.attention_weights.view(self.hidden_dims, 1)

        # 双层lstm
        self.lstm_net = nn.LSTM(self.embeding, self.hidden_dims,
                                num_layers=self.rnn_layers, dropout=self.keep_dropout,
                                bidirectional=True)
        # FC层
        # self.fc_out = nn.Linear(self.hidden_dims, self.num_classes)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(self.keep_dropout),
            nn.Linear(self.hidden_dims, self.num_classes)
        )
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def attention_net_with_w(self, lstm_out, lstm_hidden, mask_sentence):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        h = h * mask_sentence

        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # m [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))

        softmax_w = F.softmax(atten_context, dim=-1)
        mask_sentence=mask_sentence.transpose(1,2)

        softmax_w = softmax_w * mask_sentence

        softmax_w_sum = softmax_w.sum(2,keepdim=True)
        softmax_w = softmax_w/softmax_w_sum

        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)

        result = context.squeeze(1)
        return result,softmax_w

    def forward(self, sen_input,ls, mask_sentence):
        # char_id = torch.from_numpy(np.array(input[0])).long()
        # pinyin_id = torch.from_numpy(np.array(input[1])).long()




        # input : [len_seq, batch_size, embedding_dim]
        sen_input = sen_input.permute(1, 0, 2)
        sen_input=pack_padded_sequence(sen_input,ls,enforce_sorted=False)
        output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_input)
        # output : [batch_size, len_seq, n_hidden * 2]
        output=pad_packed_sequence(output)[0]
       # print(output.shape)
        output = output.permute(1, 0, 2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = final_hidden_state.permute(1, 0, 2)
        # final_hidden_state = torch.mean(final_hidden_state, dim=0, keepdim=True)

        if self.attention:

            atten_out, atten_weight = self.attention_net_with_w(output, final_hidden_state, mask_sentence)
            return self.fc_out(atten_out), atten_weight

        else:

            final_hidden_state = torch.mean(final_hidden_state, dim=1, keepdim=True)
            atten_out = final_hidden_state.squeeze(1)

            return self.fc_out(atten_out),final_hidden_state


class Model(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=300, hid_dim=96,
                 step=2, dropout=0.5, word2vec=None, freeze=True,lstm_layer=2,ettention=True):
        super(Model, self).__init__()

        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), freeze, num_words)

        self.gcn = GraphLayer(in_dim, hid_dim, act=torch.tanh, dropout=dropout, step=step)

        self.textbilstm = TextBILSTM(96, num_classes, keep_dropout=dropout, hidden_dims=96, rnn_layers=lstm_layer,ettention=ettention)

        self.read = ReadoutLayer(hid_dim, num_classes, act=torch.tanh, dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, g):

        x = self.embed(g.x)


        x = self.gcn(x, g)

        mask_sentence = self.get_mask(g,g.len_inputs)

        x_sentence=self.sentence2batch(g.x_s,g.len_inputs)
        x_s=[]
        for xi, x_1 in zip(x, x_sentence):
            x_s.append(torch.index_select(xi, 0, x_1) )

        x_s = torch.stack(x_s, dim=0)

        x_s = x_s * mask_sentence

        x,atten_weight = self.textbilstm(x_s, g.len_inputs, mask_sentence)



        return x,atten_weight

    def get_mask(self, g,length):
        mask = pad_sequence([torch.ones(l) for l in length], batch_first=True).unsqueeze(-1)
        if g.x.is_cuda: mask = mask.cuda()
        return mask

    def sentence2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)

        return x