# Placeholder for model.py 

import torch
import torch.nn as nn
import math

# Assuming data.py is in the same directory or accessible in PYTHONPATH
from .data import VOCAB_SIZE, MAX_SEQ_LEN, PAD_IDX

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe is (max_len, 1, d_model), but we want (1, max_len, d_model) for batch_first
        pe = pe.transpose(0,1) # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is (batch_size, seq_len, d_model)
        # self.pe is (1, max_len, d_model)
        # We need to select the relevant part of pe: (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NonAutoregressiveTransformer(nn.Module):
    def __init__(self, 
                 vocab_size: int = VOCAB_SIZE, 
                 d_model: int = 64, 
                 nhead: int = 4, 
                 num_encoder_layers: int = 3, 
                 dim_feedforward: int = 128, 
                 dropout: float = 0.1,
                 max_seq_len: int = MAX_SEQ_LEN,
                 padding_idx: int = PAD_IDX):
        super().__init__()
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='relu',
            batch_first=True # Important for (batch, seq, feature) input
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.dropout_layer = nn.Dropout(dropout) # General dropout after embeddings + pos encoding

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize weights for linear layers and embeddings
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def _generate_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Generates a padding mask for the source sequence.
        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, seq_len).
        Returns:
            torch.Tensor: Padding mask tensor of shape (batch_size, seq_len),
                          where True indicates a padded position.
        """
        return src == self.padding_idx

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
        Returns:
            output: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        src_padding_mask = self._generate_padding_mask(src) # (batch_size, seq_len)

        # Embedding and Positional Encoding
        # src shape: (batch_size, seq_len)
        embedded_src = self.embedding(src) * math.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        pos_encoded_src = self.pos_encoder(embedded_src) # (batch_size, seq_len, d_model)
        # Dropout after embedding and positional encoding seems common, though TransformerEncoderLayer also has dropout
        # Using the self.dropout_layer here is an explicit choice. Could also rely solely on internal dropout of layers.
        # For this small model, an extra dropout might be okay or slightly redundant.
        # Let's apply it after pos_encoder for now.
        encoder_input = self.dropout_layer(pos_encoded_src)

        # Transformer Encoder
        # TransformerEncoder expects src_key_padding_mask to be (N, S)
        # where N is batch_size, S is seq_len
        # True values are ignored, False values are attended to.
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        encoder_output = self.transformer_encoder(
            encoder_input, 
            src_key_padding_mask=src_padding_mask
        ) # (batch_size, seq_len, d_model)

        # Output Linear Layer
        output = self.output_linear(encoder_output) # (batch_size, seq_len, vocab_size)
        return output

if __name__ == '__main__':
    from .data import get_batch, INT_TO_CHAR, CHAR_TO_INT, PAD_IDX # For testing

    # Hyperparameters (can be tuned)
    BATCH_SIZE = 4
    
    # Instantiate model
    model = NonAutoregressiveTransformer(
        vocab_size=VOCAB_SIZE, 
        d_model=64, 
        nhead=4, 
        num_encoder_layers=3, 
        dim_feedforward=128, 
        dropout=0.1,
        max_seq_len=MAX_SEQ_LEN,
        padding_idx=PAD_IDX
    )
    print(f"Model instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Generate a dummy batch
    print(f"\n--- Testing with a batch from data.py (batch_size={BATCH_SIZE}) ---")
    src_batch, tgt_batch = get_batch(batch_size=BATCH_SIZE) # src_batch is (BATCH_SIZE, MAX_SEQ_LEN)
    print(f"Source batch shape: {src_batch.shape}")

    # Forward pass
    try:
        logits = model(src_batch) # (BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE)
        print(f"Output logits shape: {logits.shape}")

        # Example: Print predicted character indices for the first item in the batch
        predicted_indices = torch.argmax(logits[0], dim=-1) # (MAX_SEQ_LEN)
        print(f"Predicted indices for first item in batch: {predicted_indices.tolist()}")
        
        # Convert to string to see an example
        from .data import INT_TO_CHAR
        predicted_chars = [INT_TO_CHAR.get(idx.item(), '?') for idx in predicted_indices]
        print(f"Predicted string for first item: '{' '.join(predicted_chars)}'")
        print(f"Actual input for first item:   '{' '.join([INT_TO_CHAR.get(idx.item(), '?') for idx in src_batch[0]])}'")
        print(f"Actual target for first item:  '{' '.join([INT_TO_CHAR.get(idx.item(), '?') for idx in tgt_batch[0]])}'")

    except Exception as e:
        print(f"Error during model forward pass or test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing with a simpler, manual batch ---")
    # Manual test batch
    manual_src = torch.tensor([
        [CHAR_TO_INT['1'], CHAR_TO_INT['+'], CHAR_TO_INT['2'], CHAR_TO_INT['='], PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX], # "1+2=       "
        [CHAR_TO_INT['9'], CHAR_TO_INT['9'], CHAR_TO_INT['+'], CHAR_TO_INT['1'], CHAR_TO_INT['='], PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX, PAD_IDX]  # "99+1=      "
    ], dtype=torch.long)
    print(f"Manual source batch shape: {manual_src.shape}")

    try:
        manual_logits = model(manual_src)
        print(f"Manual output logits shape: {manual_logits.shape}")
        manual_predicted_indices_0 = torch.argmax(manual_logits[0], dim=-1)
        manual_predicted_chars_0 = [INT_TO_CHAR.get(idx.item(), '?') for idx in manual_predicted_indices_0]
        print(f"Predicted string for manual item 0: '{' '.join(manual_predicted_chars_0)}'")

    except Exception as e:
        print(f"Error during model forward pass with manual batch: {e}")
        import traceback
        traceback.print_exc() 