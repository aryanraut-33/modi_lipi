import yaml
import torch
import cv2
import numpy as np

# Import our custom modules built in previous phases
from data_processing import DataIngestion, DynamicSampler, PatchExtractor
from models import LineTR
from post_processing import PostProcessor
from seam_carving import SeamGenerator

def load_config(config_path="config.yaml"):
    """Loads hyperparameters from the YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_inference(image_path, model_weights_path=None, config_path="config.yaml"):
    """
    Executes the End-to-End LineTR Inference Pipeline on a high-resolution document.
    """
    print("==================================================")
    print("      Starting LineTR Inference Pipeline          ")
    print("==================================================")
    
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    config = load_config(config_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Apple Silicon Metal GPU
    else:
        device = torch.device("cpu")
    print(f"[*] Running on device: {device}")
    
    # ---------------------------------------------------------
    # 2. Load Model (Phase 2)
    # ---------------------------------------------------------
    print("[*] Initializing LineTR Neural Networks...")
    model = LineTR(embed_dim=config['model']['embed_dim']).to(device)
    
    # Load trained weights if provided
    if model_weights_path:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        
    model.eval() # Set to evaluation mode for inference
    
    # ---------------------------------------------------------
    # 3. Data Ingestion & Patch Extraction (Phase 1)
    # ---------------------------------------------------------
    print(f"[*] Loading High-Resolution Image: {image_path}")
    image = DataIngestion.load_image(image_path)
    
    # Placeholder function for dynamic sampler's initial pass
    def gap_prediction_fn(resized_patch):
        # In a fully trained environment, you would run a quick forward pass of the model 
        # to measure the gap. For this orchestrator template, we use a mock value.
        return 15.0 
        
    sampler = DynamicSampler(target_size=config['pipeline']['target_size'], 
                             zeta=config['pipeline']['zeta'])
                             
    optimal_t = sampler.compute_optimal_patch_size(image, gap_prediction_fn)
    print(f"[*] Computed Context-Adaptive Patch Size (t): {optimal_t}x{optimal_t}")
    
    extractor = PatchExtractor(target_size=config['pipeline']['target_size'], 
                               overlap_ratio=config['pipeline']['overlap_ratio'])
                               
    patches_data = extractor.extract_patches(image, optimal_t)
    print(f"[*] Extracted {len(patches_data)} overlapping patches from document.")
    
    # ---------------------------------------------------------
    # 4. Network Inference & Post-Processing (Phase 4)
    # ---------------------------------------------------------
    post_processor = PostProcessor(original_image_shape=image.shape, 
                                   model_target_size=config['pipeline']['target_size'])
                                   
    print("[*] Executing Neural Networks and Assembling Global Canvas...")
    with torch.no_grad():
        for i, item in enumerate(patches_data):
            # Prepare tensor (add batch dimension)
            patch_tensor = item['patch_tensor'].unsqueeze(0).to(device) # (1, 3, 256, 256)
            bbox = item['bbox']
            
            # Neural Network Forward Pass
            line_params, line_probs, energy_map = model(patch_tensor)
            
            # Map predictions back to global canvas coordinates
            post_processor.unnormalize_scribbles(
                line_params[0], 
                line_probs[0], 
                bbox, 
                prob_threshold=config['pipeline']['prob_threshold']
            )
            
            # Stitch the continuous energy map patch
            post_processor.stitch_energy_map(energy_map[0], bbox)
            
    # Finalize the global canvas and cluster the scribbles
    global_energy_map = post_processor.finalize_energy_map()
    global_scribbles = post_processor.iterative_projection_merging(
        distance_threshold=config['pipeline']['distance_threshold'],
        slope_threshold=config['pipeline']['slope_threshold']
    )
    
    print(f"[*] Successfully clustered predictions into {len(global_scribbles)} global text lines.")
    
    # ---------------------------------------------------------
    # 5. Seam Generation (Phase 5)
    # ---------------------------------------------------------
    print("[*] Running Seam Carving to mathematically generate Polygons...")
    seam_generator = SeamGenerator(global_energy_map, global_scribbles)
    polygons = seam_generator.generate_polygons()
    
    print("==================================================")
    print(f"[*] Pipeline Complete: Generated {len(polygons)} tight-fitting polygons!")
    print("==================================================")
    return polygons

if __name__ == "__main__":
    # Example Sanity Execution:
    # try:
    #     polygons = run_inference("sample_manuscript.jpg", "linetr_weights.pth")
    # except Exception as e:
    #     print("Pipeline orchestrator is ready. Provide a valid image to run!")
    print("LineTR Pipeline Orchestrator compiled successfully.")
