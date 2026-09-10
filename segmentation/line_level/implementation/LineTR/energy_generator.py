import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """ 
    Downsamples the 256x256 image patch to a 16x16 feature map. 
    Maintains skip connections for spatial reconstruction later. 
    """
    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        # 4 reduction layers: 256 -> 128 -> 64 -> 32 -> 16
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)
        self.enc4 = self._block(128, embed_dim)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        f1 = self.enc1(x)  # (B, 32, 128, 128)
        f2 = self.enc2(f1) # (B, 64, 64, 64)
        f3 = self.enc3(f2) # (B, 128, 32, 32)
        p  = self.enc4(f3) # (B, 256, 16, 16) Feature map P
        return p, [f1, f2, f3]

class CNNDecoder(nn.Module):
    """
    Upsamples the transformer output back to 256x256 using skip connections 
    from the CNN Encoder to retain high-frequency details.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.dec4 = self._block(embed_dim, 128)
        self.dec3 = self._block(128 + 128, 64) # +128 from skip f3
        self.dec2 = self._block(64 + 64, 32)   # +64 from skip f2
        self.dec1 = self._block(32 + 32, 16)   # +32 from skip f1
        self.final = nn.Conv2d(16, 1, kernel_size=1) # Condense to 1 channel (Energy Map)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skips):
        f1, f2, f3 = skips
        
        # Step-by-step upsampling and channel concatenation (skip connections)
        x = self.dec4(x)
        x = torch.cat([x, f3], dim=1)
        
        x = self.dec3(x)
        x = torch.cat([x, f2], dim=1)
        
        x = self.dec2(x)
        x = torch.cat([x, f1], dim=1)
        
        x = self.dec1(x)
        x = self.final(x)
        
        # Sigmoid to create continuous [0, 1] map
        return torch.sigmoid(x)

class TextEnergyMapGenerator(nn.Module):
    """
    Hybrid CNN-Transformer network that generates a continuous [0, 1] heat map
    indicating the density/presence of text lines in the patch.
    """
    def __init__(self, embed_dim=256, depth=3, num_heads=8):
        super().__init__()
        self.cnn_encoder = CNNEncoder(in_channels=3, embed_dim=embed_dim)
        
        # Transformer Decoder (R=3 layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            activation="gelu", 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        self.cnn_decoder = CNNDecoder(embed_dim=embed_dim)

    def forward(self, image_patch, vit_memory):
        """
        Args:
            image_patch: (B, 3, 256, 256) The raw extracted image patch
            vit_memory: (B, 256, 16, 16) The output map M from the ViT Backbone
            
        Returns:
            energy_map: (B, 1, 256, 256) Continuous map of text density
        """
        B = image_patch.shape[0]
        
        # 1. Get CNN features 'P' and skips for later reconstruction
        p, skips = self.cnn_encoder(image_patch) # p: (B, 256, 16, 16)
        
        # 2. Transformer Cross-Attention Phase
        # Target sequence is the CNN features 'P'
        tgt = p.flatten(2).transpose(1, 2) # (B, 256, embed_dim)
        
        # Memory sequence is the ViT features 'M'
        memory = vit_memory.flatten(2).transpose(1, 2) # (B, 256, embed_dim)
        
        # Decoder self-attends on CNN targets, and cross-attends to ViT memory
        attn_out = self.transformer_decoder(tgt, memory) # (B, 256, embed_dim)
        
        # Reshape the output back to a spatial 16x16 feature map
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, 16, 16)
        
        # 3. CNN Decoder reconstructions
        energy_map = self.cnn_decoder(attn_out, skips) # (B, 1, 256, 256)
        
        return energy_map
