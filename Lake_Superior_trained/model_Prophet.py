import torch
import numpy as np

# Define a configuration class for Transformer parameters
class Transconfig():
    def __init__(self):
        self.d_model = 7          # Dimension of model input features -- default
        self.d_ff = 2048          # Dimension of feed-forward network
        self.d_k = self.d_v = 128 # Dimension of key (K) and value (V) in attention
        self.n_layers = 6         # Number of encoder layers
        self.n_heads = 8          # Number of attention heads
        self.fc_p = 128           # Fully connected layer size in the prediction layer
        self.output_size = 1      # Output dimension
        self.batch_size = 256     # Batch size
        self.cycle_num = 12       # Number of time steps in input sequence -- default

# Initialize transformer configuration
transconfig = Transconfig()


# Positional Encoding for sequence representation
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=transconfig.d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encoding table
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)
        ])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # Apply sine to even indices
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # Apply cosine to odd indices
        self.pos_table = torch.FloatTensor(pos_table)  # Store as a tensor

    def forward(self, enc_inputs):
        # Add positional encoding to the input sequence
        enc_inputs = enc_inputs + self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


# Scaled Dot-Product Attention Mechanism
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(transconfig.d_k)  # Scale by sqrt(d_k)
        attn = torch.nn.Softmax(dim=-1)(scores)  # Apply softmax along the last dimension
        context = torch.matmul(attn, V)  # Multiply with value matrix V

        return context, attn


# Multi-Head Attention Layer
class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        
        # Linear transformations for Q, K, and V
        self.W_Q = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_K = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_V = torch.nn.Linear(transconfig.d_model, transconfig.d_v * transconfig.n_heads, bias=False)
        self.fc = torch.nn.Linear(transconfig.n_heads * transconfig.d_v, transconfig.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)

        # Compute Q, K, V matrices and reshape for multi-head attention
        Q = self.W_Q(input_Q).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, transconfig.n_heads, transconfig.d_v).transpose(1, 2)

        # Compute attention
        context, attn = ScaledDotProductAttention()(Q, K, V)

        # Concatenate attention outputs from all heads
        context = context.transpose(1, 2).reshape(batch_size, -1, transconfig.n_heads * transconfig.d_v)
        output = self.fc(context)

        # Apply residual connection and layer normalization
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual), attn


# Position-wise Feed-Forward Network
class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transconfig.d_model, transconfig.d_ff, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.d_ff, transconfig.d_model, bias=False))

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual)


# Transformer Encoder Layer
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # Multi-head attention layer
        self.pos_ffn = PoswiseFeedForwardNet()  # Feed-forward network

    def forward(self, enc_inputs):
        # Self-attention and feed-forward processing
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


# Full Transformer Encoder
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(transconfig.d_model)  # Add positional encoding
        self.layers = torch.nn.ModuleList([EncoderLayer() for _ in range(transconfig.n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs)  # Apply positional encoding
        enc_self_attns = []

        # Pass through each encoder layer
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs, enc_self_attns


# Prediction Layer for Final Output
class PredictLayer(torch.nn.Module):
    def __init__(self):
        super(PredictLayer, self).__init__()
        
        self.fc_fin = torch.nn.Sequential(
            torch.nn.Linear(transconfig.cycle_num * transconfig.d_model, transconfig.fc_p, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.fc_p, transconfig.output_size, bias=False)) 
        
    def forward(self, enc_outputs):
        output = self.fc_fin(enc_outputs)
        return output


# Transformer Model for Sequence Prediction
class Transformer_half_base(torch.nn.Module):
    def __init__(self, parameter_list):
        super(Transformer_half_base, self).__init__()
        transconfig.cycle_num = parameter_list[0]  # Update cycle number
        self.Encoder = Encoder()
        self.Predict = PredictLayer()
        
    def forward(self, enc_inputs, parameter_list):
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)  # Encode input sequence
        enc_outputs = enc_outputs.reshape(-1, transconfig.cycle_num * transconfig.d_model)  # Reshape for prediction
        outputs = self.Predict(enc_outputs)  # Final prediction

        return outputs, enc_self_attns