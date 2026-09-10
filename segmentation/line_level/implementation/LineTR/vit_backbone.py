import torch
import torch.nn as nn

class ViTBackbone(nn.Module):
    """
    Vision Transformer Backbone for LineTR.
    Splits a 256x256 image patch into 16x16 patches and processes them.
    Output feature map M has dimension 16x16 with an embedding dimension of 256.
    """
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=256, depth=8, num_heads=8):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Patch embedding: Convolution with stride=patch_size behaves like non-overlapping linear projections
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable position embeddings (1D sequence format matching the flattened grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        """
        Args:
            x (Tensor): Raw image patches of shape (B, 3, 256, 256)
            
        Returns:
            Tensor: Shared feature map M of shape (B, 256, 16, 16)
        """
        B = x.shape[0]
        
        # 1. Patch embedding mapping: (B, 256, 16, 16)
        x = self.patch_embed(x)
        
        # 2. Flatten spatial dimensions for Transformer: (B, 256, 256) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # 3. Add positional embeddings to retain spatial awareness
        x = x + self.pos_embed
        
        # 4. Pass sequence through Transformer Encoder
        x = self.transformer(x)
        
        # 5. Reshape back to spatial feature map: (B, embed_dim, 16, 16)
        x = x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        
        return x
