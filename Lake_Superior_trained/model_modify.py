import torch
import numpy as np

#Critic!

# Configuration class for Critic parameters
class Transconfig():
    def __init__(self):
        self.d_model = 6  # Dimension of input features--default
        self.d_ff = 2048  # Dimension of feed-forward network
        self.d_k = self.d_v = 128  # Dimension of key and value vectors
        self.n_layers = 6  # Number of encoder layers
        self.n_heads = 8  # Number of attention heads
        self.fc_p = 128  # Fully connected layer dimension
        self.output_size = 1  # Output dimension--default
        self.cycle_num = 5  # Number of time steps in input sequence (future)--default

# Initialize Transformer configuration
transconfig = Transconfig()

# Positional Encoding for adding sequential order information
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=transconfig.d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)
        ])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # Apply sine to even indices
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # Apply cosine to odd indices
        self.pos_table = torch.FloatTensor(pos_table)

    def forward(self, enc_inputs):
        # Add positional encoding to input tensor
        enc_inputs = enc_inputs + self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)

# Scaled Dot-Product Attention
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        # Compute scaled attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(transconfig.d_k)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

# Multi-Head Attention Layer
class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_K = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_V = torch.nn.Linear(transconfig.d_model, transconfig.d_v * transconfig.n_heads, bias=False)
        self.fc = torch.nn.Linear(transconfig.n_heads * transconfig.d_v, transconfig.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        batch_size = input_Q.size(0)
        
        # Transform inputs into multiple attention heads
        Q = self.W_Q(input_Q).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, transconfig.n_heads, transconfig.d_v).transpose(1, 2)
        
        # Apply scaled dot-product attention
        context, attn = ScaledDotProductAttention()(Q, K, V)
        
        # Concatenate heads and pass through linear layer
        context = context.transpose(1, 2).reshape(batch_size, -1, transconfig.n_heads * transconfig.d_v)
        output = self.fc(context)
        return torch.nn.LayerNorm(transconfig.d_model)(output + input_Q), attn

# Position-wise Feed-Forward Network
class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transconfig.d_model, transconfig.d_ff, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.d_ff, transconfig.d_model, bias=False)
        )

    def forward(self, inputs):
        return torch.nn.LayerNorm(transconfig.d_model)(self.fc(inputs) + inputs)

# Transformer Encoder Layer
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# Transformer Encoder
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(transconfig.d_model)
        self.layers = torch.nn.ModuleList([EncoderLayer() for _ in range(transconfig.n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# Prediction Layer
class PredictLayer(torch.nn.Module):
    def __init__(self):
        super(PredictLayer, self).__init__()
        self.fc_fin = torch.nn.Sequential(
            torch.nn.Linear(transconfig.cycle_num * transconfig.d_model, transconfig.fc_p, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.fc_p, transconfig.output_size, bias=False)
        )

    def forward(self, enc_outputs):
        return self.fc_fin(enc_outputs)

# Transformer Model
class Transformer_half_modify(torch.nn.Module):
    def __init__(self, parameter):
        super(Transformer_half_modify, self).__init__()
        self.Encoder = Encoder()
        self.Predict = PredictLayer()
        transconfig.d_model = parameter.input_size - 1
        transconfig.cycle_num = parameter.cycle_num_later
        transconfig.output_size = parameter.output_size

    def forward(self, enc_inputs):
        enc_inputs = torch.FloatTensor(enc_inputs).reshape((-1, 5, 6))
        enc_outputs, _ = self.Encoder(enc_inputs)
        enc_outputs = enc_outputs.reshape(-1, transconfig.cycle_num * transconfig.d_model)
        outputs = self.Predict(enc_outputs)
        return outputs
