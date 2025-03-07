import torch
import numpy as np

# Define a configuration class for Transformer parameters
class Transconfig():
    def __init__(self):
        self.d_model = 9          # Dimension of model input features -- default
        self.d_ff = 2048          # Dimension of feed-forward network
        self.d_k = self.d_v = 128 # Dimension of key (K) and value (V) in attention
        self.n_layers = 6         # Number of encoder layers
        self.n_heads = 8          # Number of attention heads
        self.fc_p = 128           # Fully connected layer size in the prediction layer
        self.output_size = 1      # Output dimension
        self.batch_size = 256     # Batch size
        self.cycle_num = 12       # Number of time steps in input sequence -- default

# Instantiate the configuration
transconfig = Transconfig()


# Positional Encoding module to incorporate positional information into embeddings
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=transconfig.d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Generate positional encoding table
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)
        ])

        # Apply sine to even indices and cosine to odd indices
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])

        # Convert to a tensor
        self.pos_table = torch.FloatTensor(pos_table)

    def forward(self, enc_inputs):
        # Add positional encoding to input embeddings
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


# Scaled Dot-Product Attention mechanism
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):  
        # Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(transconfig.d_k)  
        attn = torch.nn.Softmax(dim=-1)(scores)  # Apply softmax
        context = torch.matmul(attn, V)  # Compute weighted sum of values
        return context, attn  # Return context and attention weights


# Multi-Head Attention mechanism
class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        # Linear transformation layers for Q, K, V
        self.W_Q = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_K = torch.nn.Linear(transconfig.d_model, transconfig.d_k * transconfig.n_heads, bias=False)
        self.W_V = torch.nn.Linear(transconfig.d_model, transconfig.d_v * transconfig.n_heads, bias=False)
        self.fc = torch.nn.Linear(transconfig.n_heads * transconfig.d_v, transconfig.d_model, bias=False)  # Output layer

    def forward(self, input_Q, input_K, input_V):
        # Save residual connection
        residual, batch_size = input_Q, input_Q.size(0)

        # Linear projection and reshaping for multi-head mechanism
        Q = self.W_Q(input_Q).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, transconfig.n_heads, transconfig.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, transconfig.n_heads, transconfig.d_v).transpose(1, 2)

        # Apply scaled dot-product attention
        context, attn = ScaledDotProductAttention()(Q, K, V)

        # Reshape and apply final linear transformation
        context = context.transpose(1, 2).reshape(batch_size, -1, transconfig.n_heads * transconfig.d_v)
        output = self.fc(context)

        # Apply residual connection and layer normalization
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual), attn


# Position-wise Feedforward Network
class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(transconfig.d_model, transconfig.d_ff, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.d_ff, transconfig.d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs  # Save residual connection
        output = self.fc(inputs)  # Apply feedforward layers
        return torch.nn.LayerNorm(transconfig.d_model)(output + residual)  # Residual connection and layer normalization


# Encoder Layer: consists of Multi-Head Attention and Feedforward Network
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # Self-attention mechanism
        self.pos_ffn = PoswiseFeedForwardNet()  # Feedforward network

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # Self-attention
        enc_outputs = self.pos_ffn(enc_outputs)  # Feedforward network
        return enc_outputs, attn  # Return encoded outputs and attention weights


# Transformer Encoder: Stack of Encoder Layers
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(transconfig.d_model)  # Positional encoding
        self.layers = torch.nn.ModuleList([EncoderLayer() for _ in range(transconfig.n_layers)])  # Stacking encoder layers

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs)  # Add positional encoding
        enc_self_attns = []

        # Pass through each encoder layer
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)  # Store attention weights

        return enc_outputs, enc_self_attns  # Return final encoded outputs and attention weights


# Prediction Layer: Fully connected network for final output prediction
class PredictLayer(torch.nn.Module):
    def __init__(self):
        super(PredictLayer, self).__init__()
        
        self.fc_fin = torch.nn.Sequential(
            torch.nn.Linear(transconfig.cycle_num * transconfig.d_model, transconfig.fc_p, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(transconfig.fc_p, transconfig.output_size, bias=False)
        )

    def forward(self, enc_outputs):
        return self.fc_fin(enc_outputs)  # Compute the final prediction


# Transformer Model with Encoder and Prediction Layer
class Transformer_half_base(torch.nn.Module):
    def __init__(self, parameter_list, parameter):
        super(Transformer_half_base, self).__init__()

        # Update configuration based on input parameters
        transconfig.cycle_num = parameter_list[0]
        transconfig.d_model = parameter.input_size

        # Initialize Encoder and Prediction Layer
        self.Encoder = Encoder()
        self.Predict = PredictLayer()

    def forward(self, enc_inputs, parameter_list):
        # Encode input sequences
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)

        # Reshape encoder outputs for prediction
        enc_outputs = enc_outputs.reshape(transconfig.batch_size, transconfig.cycle_num * transconfig.d_model)

        # Compute final predictions
        outputs = self.Predict(enc_outputs)

        return outputs, enc_self_attns  # Return predicted outputs and attention weights