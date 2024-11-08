import torch
import numpy as np

class Transconfig():
    def __init__(self):
        self.d_model = 7
        self.d_ff = 2048
        self.d_k = self.d_v = 128
        self.n_layers = 6
        self.n_heads = 8
        self.fc_p = 128
        self.output_size = 1
        self.batch_size = 256
        self.cycle_num = 12

transconfig = Transconfig()


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model = transconfig.d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2*i/d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)        # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):
                                                              # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):                                         # Q: [batch_size, n_heads, len_q, d_k]
                                                                        # K: [batch_size, n_heads, len_k, d_k]
                                                                        # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(transconfig.d_k)    # scores : [batch_size, n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                 # [batch_size, n_heads, len_q, d_v]

        return context, attn
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_K = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_V = torch.nn.Linear(transconfig.d_model, transconfig.d_v * transconfig.n_heads, bias=False)
        self.fc = torch.nn.Linear(transconfig.n_heads * transconfig.d_v, transconfig.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):               # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, transconfig.n_heads, transconfig.d_v).transpose(1, 2)    # V: [batch_size, n_heads, len_v(=len_k), d_v]                                                                                                                
        context, attn = ScaledDotProductAttention()(Q, K, V)                        # context: [batch_size, n_heads, len_q, d_v]
                                                                                    # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, transconfig.n_heads * transconfig.d_v)
                                                                                    # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                   # [batch_size, len_q, d_model]
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual), attn


class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transconfig.d_model, transconfig.d_ff, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.d_ff, transconfig.d_model, bias=False))

    def forward(self, inputs):                                  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                   # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                      # 前馈神经网络

    def forward(self, enc_inputs):              # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
                                                                    # enc_outputs: [batch_size, src_len, d_model],
                                                                    # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(transconfig.d_model)                               # 加入位置信息
        self.layers = torch.nn.ModuleList([EncoderLayer() for _ in range(transconfig.n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        enc_outputs = self.pos_emb(enc_inputs)                                    # enc_outputs: [batch_size, src_len, d_model]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)                      # enc_outputs :   [batch_size, src_len, d_model]
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs, enc_self_attns


class PredictLayer(torch.nn.Module):
    def __init__(self):
        super(PredictLayer, self).__init__()
        
        self.fc_fin = torch.nn.Sequential(
            torch.nn.Linear(transconfig.cycle_num*transconfig.d_model, transconfig.fc_p, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.fc_p, transconfig.output_size, bias=False)) 
        
    def forward(self, enc_outputs):
        
        output = self.fc_fin(enc_outputs)
        
        return output


class Transformer_half_base(torch.nn.Module):
    def __init__(self, parameter_list):
        super(Transformer_half_base, self).__init__()
        self.Encoder = Encoder()
        transconfig.cycle_num = parameter_list[0]
        self.Predict = PredictLayer()
        
    
    def forward(self, enc_inputs, parameter_list):                                      # enc_inputs: [batch_size, src_len]
                                                                        # dec_inputs: [batch_size, tgt_len]

        
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)          # enc_outputs: [batch_size, src_len, d_model],
                                                                        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs = enc_outputs.reshape(transconfig.batch_size, transconfig.cycle_num*transconfig.d_model)  #应注意
        outputs = self.Predict(enc_outputs)

        return outputs, enc_self_attns