import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader

# Import our custom architecture modules
from data_processing import PatchExtractor, LineTRDataset
from models import LineTR
from losses import LineTRLoss, TextEnergyLoss
from trainer import LineTRTrainer

def get_cbad_files(dataset_path, split="train"):
    """
    Finds all .jpg and corresponding .xml files in the dataset folder.
    Note: The user specifically requested using 'test' folder for training 
    since it contains the largest amount of data (~1500 images).
    """
    image_dir = os.path.join(dataset_path, split)
    xml_dir = os.path.join(dataset_path, split, "page")
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    xml_paths = []
    valid_image_paths = []
    
    for img_path in image_paths:
        # The XML file shares the same basename as the JPG
        basename = os.path.basename(img_path).replace(".jpg", ".xml")
        xml_path = os.path.join(xml_dir, basename)
        
        if os.path.exists(xml_path):
            valid_image_paths.append(img_path)
            xml_paths.append(xml_path)
            
    return valid_image_paths, xml_paths

def main():
    parser = argparse.ArgumentParser(description="LineTR Training Pipeline")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Test the dataloader by fetching one batch and exiting. Does not initialize models or consume VRAM.")
    args = parser.parse_args()

    print("==================================================")
    print("      LineTR Deep Learning Training Pipeline      ")
    print("==================================================")
    
    # Dynamically resolve dataset path so it works perfectly locally or in Colab
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_dir, "dataset", "CBAD_dataset")
    
    # ---------------------------------------------------------
    # 1. Load File Paths
    # ---------------------------------------------------------
    print("[*] Finding dataset files...")
    # Swap splits based on user instruction (use test for training)
    train_imgs, train_xmls = get_cbad_files(dataset_root, split="test")
    eval_imgs, eval_xmls = get_cbad_files(dataset_root, split="eval")
    
    print(f"[*] Found {len(train_imgs)} image-annotation pairs for Training.")
    
    if len(train_imgs) == 0:
        print("[!] Error: No training data found. Check dataset path.")
        return
        
    # Note: We are loading the full dataset. Ensure you have enough VRAM on Colab.
    # train_imgs = train_imgs[:5]
    # train_xmls = train_xmls[:5]
    
    # ---------------------------------------------------------
    # 2. Initialize DataLoader
    # ---------------------------------------------------------
    print("[*] Building PyTorch DataLoader (Extracting patches & parsing XMLs)...")
    extractor = PatchExtractor(target_size=256, overlap_ratio=0.25)
    
    # optimal_patch_size=None tells the dataset to calculate Context-Adaptive patches dynamically per image!
    train_dataset = LineTRDataset(image_paths=train_imgs, 
                                  patch_extractor=extractor, 
                                  optimal_patch_size=None, 
                                  xml_paths=train_xmls)
                                  
    # Crucial: Use our custom collate_fn to handle variable lengths of ground-truth lines
    # Note: shuffle=False is required because LineTRDataset is an IterableDataset that handles its own internal shuffling.
    # Increased batch_size to 128 to fully saturate the 15GB Colab GPU VRAM!
    # pin_memory=True accelerates the transfer of tensors from the CPU to the GPU.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, 
                              pin_memory=True, collate_fn=LineTRDataset.custom_collate_fn)
                              
    print(f"[*] DataLoader ready. Yields {len(train_dataset)} total training patches.")
    
    # Create Validation Loader
    print("[*] Building Validation DataLoader...")
    eval_dataset = LineTRDataset(image_paths=eval_imgs, 
                                 patch_extractor=extractor, 
                                 optimal_patch_size=None, 
                                 xml_paths=eval_xmls)
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, 
                             pin_memory=True, collate_fn=LineTRDataset.custom_collate_fn)
    
    # ---------------------------------------------------------
    # 2.5 Optional Dry Run (Verify Dataloader & RAM)
    # ---------------------------------------------------------
    if args.dry_run:
        print("\n" + "="*50)
        print(">> EXECUTING DRY RUN: Fetching a single batch from DataLoader")
        print("==================================================")
        try:
            for batch_idx, (images, gt_lines, gt_energy_maps) in enumerate(train_loader):
                print(f"Batch {batch_idx + 1} successfully loaded!")
                print(f" -> Image Tensor Shape: {images.shape} (Batch, Channels, Height, Width)")
                print(f" -> Energy Map Tensor Shape: {gt_energy_maps.shape} (Batch, Channels, Height, Width)")
                print(f" -> Ground Truth Lines: A list of {len(gt_lines)} tensors containing line parameters.")
                for i, lines in enumerate(gt_lines):
                    print(f"    - Patch {i} contains {lines.shape[0]} text lines.")
                print("\n[SUCCESS] Dataset parsed and loaded perfectly! VRAM was not touched.")
                break # Just test the first batch
        except Exception as e:
            print(f"\n[ERROR] Dataloader crashed during dry run: {e}")
            
        print("[*] Exiting dry run.")
        return # Exit the script completely so training never starts
    
    # ---------------------------------------------------------
    # 3. Initialize Models and Losses
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # Apple Silicon Metal GPU
    else:
        device = torch.device("cpu")
    print(f"[*] Initializing Models on {device}...")
    
    model = LineTR(embed_dim=256)
    line_loss = LineTRLoss()
    energy_loss = TextEnergyLoss()
    
    trainer = LineTRTrainer(model, line_loss, energy_loss, device=device)
    
    # ---------------------------------------------------------
    # 4. Execute Two-Phase Training
    # ---------------------------------------------------------
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    print("\n" + "="*50)
    print(">> INITIATING PHASE 1: Train Backbone & Line Generator")
    print("==================================================")
    
    # If Colab crashes, change resume_from to the last saved epoch file!
    # Example: resume_from = os.path.join(checkpoint_dir, "linetr_phase1_epoch_10.pth")
    trainer.train_phase_1(train_loader, epochs_part1=60, epochs_part2=20, 
                          checkpoint_dir=checkpoint_dir, resume_from=None,
                          eval_loader=eval_loader)
    
    print("\n" + "="*50)
    print(">> INITIATING PHASE 2: Train Text-Energy Map Generator")
    print("==================================================")
    
    # Phase 2 automatically starts after Phase 1 finishes!
    trainer.train_phase_2(train_loader, epochs=50, 
                          checkpoint_dir=checkpoint_dir, resume_from=None,
                          eval_loader=eval_loader)
    
    print("\n[*] Training Pipeline Orchestrator compiled successfully!")

if __name__ == "__main__":
    main()
