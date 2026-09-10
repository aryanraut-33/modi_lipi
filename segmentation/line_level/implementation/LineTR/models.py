import torch
import torch.nn as nn
from vit_backbone import ViTBackbone
from line_generator import LineParameterGenerator
from energy_generator import TextEnergyMapGenerator

class LineTR(nn.Module):
    """
    The Orchestrator. 
    Complete LineTR Stage-1 Architecture integrating the ViT Backbone, 
    Line-Parameter Generator, and Text-Energy Map Generator.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # 1. The Eye
        self.backbone = ViTBackbone(embed_dim=embed_dim)
        
        # 2. The Ruler
        self.line_generator = LineParameterGenerator(embed_dim=embed_dim)
        
        # 3. The Highlighter
        self.energy_generator = TextEnergyMapGenerator(embed_dim=embed_dim)

    def forward(self, image_patch):
        """
        Executes a single forward pass of the entire Stage-1 pipeline.
        
        Args:
            image_patch: Tensor of shape (B, 3, 256, 256)
            
        Returns:
            line_params: (B, 200, 3) The predicted scribbles (mu_x, mu_y, m)
            line_probs: (B, 200, 1) The confidence scores for each scribble
            energy_map: (B, 1, 256, 256) The continuous text-energy map [0, 1]
        """
        
        # Phase 2A: Shared Encoder computes memory representation M
        memory_m = self.backbone(image_patch)
        
        # Phase 2B: Line-Parameter Generator outputs mathematical scribbles
        line_params, line_probs = self.line_generator(memory_m)
        
        # Phase 2C: Text-Energy Map Generator outputs the continuous map
        energy_map = self.energy_generator(image_patch, memory_m)
        
        return line_params, line_probs, energy_map

# --------------------------------------------------------------------------------
# Example Sanity Check Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Initializing Phase 2: Core Network Architecture (Stage-1 Models)")
    
    # Instantiate the full LineTR model
    model = LineTR(embed_dim=256)
    
    # Create a dummy batch of 2 extracted patches (from Phase 1)
    dummy_patches = torch.randn(2, 3, 256, 256)
    
    print(f"Feeding batch of patches {dummy_patches.shape} into LineTR...")
    
    # Run the model
    with torch.no_grad():
        lines, probs, energy = model(dummy_patches)
        
    print("\n[SUCCESS] Model Outputs Generated:")
    print(f"1. Line Geometries Shape: {lines.shape} -> (Batch, Queries, mu_x/mu_y/m)")
    print(f"2. Line Probabilities Shape: {probs.shape} -> (Batch, Queries, confidence)")
    print(f"3. Text-Energy Map Shape: {energy.shape} -> (Batch, Channel, Height, Width)")
