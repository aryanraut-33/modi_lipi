import torch
import torch.nn as nn

class LineParameterGenerator(nn.Module):
    """
    DETR-style network for generating parametric representations of scribbles.
    Instead of outputting pixel blobs, it outputs geometric coordinates.
    """
    def __init__(self, embed_dim=256, num_queries=200, depth=8, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        
        # Line Priors: N horizontal lines uniformly distributed
        # In a full decoupled implementation, these priors inform positional queries.
        # Here we initialize learnable query embeddings representing our N lines.
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # Transformer Decoder Stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation="gelu", 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        # Prediction Heads (Feed-Forward Networks)
        # Head 1: Predicts line geometry -> (mu_x, mu_y, m)
        self.line_params_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3) 
        )
        
        # Head 2: Predicts confidence score for each line -> (probability)
        self.prob_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1) 
        )

    def forward(self, memory):
        """
        Args:
            memory (Tensor): Output feature map M from the ViT backbone. Shape: (B, 256, 16, 16)
            
        Returns:
            line_params (Tensor): Predicted geometries (B, 200, 3)
            probs (Tensor): Confidence scores (B, 200, 1)
        """
        B = memory.shape[0]
        
        # 1. Flatten memory feature map for the Transformer: (B, 256, embed_dim)
        memory_flat = memory.flatten(2).transpose(1, 2)
        
        # 2. Initialize sequence of line queries as zeros: (B, num_queries, embed_dim)
        tgt = torch.zeros(B, self.num_queries, memory.shape[1], device=memory.device)
        
        # 3. Get query embeddings (positional information): (B, num_queries, embed_dim)
        query_pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        
        # Note: In standard DETR, tgt is added to query_pos internally in the decoder block.
        # We manually add it here to kickstart the query sequence before feeding it to standard PyTorch Decoder.
        tgt = tgt + query_pos
        
        # 4. Pass through Transformer Decoder
        # Cross-attends line queries (tgt) with image features (memory_flat)
        hs = self.transformer_decoder(tgt, memory_flat)
        
        # 5. Extract geometric line parameters and confidence bounds
        line_params = self.line_params_head(hs) # (B, num_queries, 3)
        
        # Apply Sigmoid specifically to mu_x, mu_y to keep them within [0, 1] boundaries. 
        # The slope 'm' is unbounded, so it remains as is.
        line_params[..., 0:2] = torch.sigmoid(line_params[..., 0:2])
        
        # Extract and constrain probabilities
        prob_logits = self.prob_head(hs) # (B, num_queries, 1)
        probs = torch.sigmoid(prob_logits)
        
        return line_params, probs
