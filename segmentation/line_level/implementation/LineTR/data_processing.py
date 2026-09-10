import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as transforms
import random
import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Suppress DecompressionBombWarning for historical datasets

class DataIngestion:
    """
    Handles loading and basic preprocessing of high-resolution manuscript images.
    """
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Loads an image from a given path and normalizes its color channels.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            np.ndarray: RGB image array.
        """
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        # Convert from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class DynamicSampler:
    """
    Implements the context-adaptive patching logic. Calculates the optimal patch size (t)
    for a document based on the interline gap estimated by a preliminary model forward-pass.
    """
    def __init__(self, target_size: int = 256, zeta: float = 3.0):
        """
        Args:
            target_size (int): The size the model expects (e.g., 256x256).
            zeta (float): Scaling factor representing the expected number of text lines per patch.
        """
        self.target_size = target_size
        self.zeta = zeta
        self.candidate_sizes = [64, 128, 256, 384, 512]

    def compute_optimal_patch_size(self, image: np.ndarray, model_predict_gap_fn) -> int:
        """
        Extracts candidate patches, uses a model to predict the interline gap, 
        and computes the optimal patch size for the whole document.
        
        Args:
            image (np.ndarray): The full high-resolution image.
            model_predict_gap_fn (callable): A function that takes a batch of 256x256 
                                             patches and returns a list of predicted median 
                                             interline gaps (delta_patch).
                                             
        Returns:
            int: The optimal context-adapted patch size (t).
        """
        h, w, _ = image.shape
        estimated_doc_gaps = []

        for m in self.candidate_sizes:
            # Skip if the candidate size is larger than the image itself
            if m > h or m > w:
                continue
                
            # Extract a candidate patch (e.g., from the center of the image)
            center_y, center_x = h // 2, w // 2
            half_m = m // 2
            
            # Ensure boundaries are respected
            y1 = max(0, center_y - half_m)
            y2 = min(h, center_y + half_m)
            x1 = max(0, center_x - half_m)
            x2 = min(w, center_x + half_m)
            
            patch = image[y1:y2, x1:x2]
            
            # Resize candidate patch to the network's target size (256x256)
            patch_resized = cv2.resize(patch, (self.target_size, self.target_size))
            
            # Prepare for model inference (normalize, add batch dimension, etc.)
            # Note: This interfaces with Phase 2 models.
            delta_patch = model_predict_gap_fn(patch_resized)
            
            # Scale the patch-level gap back to the document-level gap
            delta_doc = delta_patch * (m / self.target_size)
            estimated_doc_gaps.append(delta_doc)
            
        if not estimated_doc_gaps:
            # Fallback if image is too small
            return min(h, w)
            
        # Average the estimated document gaps
        delta_avg = sum(estimated_doc_gaps) / len(estimated_doc_gaps)
        
        # Calculate optimal patch size: t = delta_avg * zeta
        t = int(delta_avg * self.zeta)
        
        # Ensure patch size is reasonable
        t = max(64, min(t, min(h, w)))
        return t

class PatchExtractor:
    """
    Handles the sliding-window grid generation and patch extraction.
    """
    def __init__(self, target_size: int = 256, overlap_ratio: float = 0.25):
        """
        Args:
            target_size (int): The size to resize extracted patches to (e.g., 256x256).
            overlap_ratio (float): How much patches should overlap (0.0 to 1.0).
        """
        self.target_size = target_size
        self.overlap_ratio = overlap_ratio

    def extract_patches(self, image: np.ndarray, patch_size: int):
        """
        Extracts overlapping patches of size `patch_size` from the image and resizes them.
        
        Args:
            image (np.ndarray): Full high-resolution image.
            patch_size (int): The optimal patch size (t) computed by DynamicSampler.
            
        Returns:
            list of dict: Contains the resized patch tensor and its original bounding box.
        """
        h, w, _ = image.shape
        stride = int(patch_size * (1 - self.overlap_ratio))
        
        # Ensure stride is at least 1
        stride = max(1, stride)
        
        patches_data = []
        
        # Sliding window loops
        for y in range(0, h - patch_size + stride, stride):
            for x in range(0, w - patch_size + stride, stride):
                # Adjust boundaries if the window exceeds image dimensions
                y1 = min(y, h - patch_size)
                y2 = y1 + patch_size
                x1 = min(x, w - patch_size)
                x2 = x1 + patch_size
                
                # Extract patch
                patch = image[y1:y2, x1:x2]
                
                # Resize to model input size uniformly
                patch_resized = cv2.resize(patch, (self.target_size, self.target_size))
                
                # Convert to tensor and normalize (standard ImageNet normalization)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])
                patch_tensor = transform(patch_resized)
                
                # Store patch along with its original coordinates for Phase 4 Stitching
                patches_data.append({
                    "patch_tensor": patch_tensor,
                    "bbox": (x1, y1, x2, y2) # Original coordinates in full image
                })
                
        return patches_data

class LineTRDataset(IterableDataset):
    """
    Lazy-loading IterableDataset for LineTR. 
    Prevents System RAM crashes by only loading ONE high-res image into memory at a time,
    yielding its patches, and then immediately clearing it before moving to the next.
    """
    def __init__(self, image_paths: list, patch_extractor: PatchExtractor, optimal_patch_size: int = None, xml_paths: list = None):
        self.image_paths = image_paths
        self.xml_paths = xml_paths
        self.patch_extractor = patch_extractor
        self.patch_size = optimal_patch_size

    def __len__(self):
        """Mathematically pre-calculates the total number of patches for tqdm progress bars."""
        total_patches = 0
        from PIL import Image
        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                w, h = img.size
            
            # For the tqdm progress bar approximation, just assume a standard size if dynamic
            p_size = self.patch_size or 256
            stride = max(1, int(p_size * (1 - self.patch_extractor.overlap_ratio)))
            n_y = len(range(0, max(1, h - p_size + stride), stride))
            n_x = len(range(0, max(1, w - p_size + stride), stride))
            total_patches += (n_y * n_x)
        return total_patches

    def __iter__(self):
        from dataset_parser import CBADParser
        
        # Shuffle the order of images for this epoch
        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        
        for idx in indices:
            img_path = self.image_paths[idx]
            
            # 1. Ingest only ONE image into RAM at a time
            image = DataIngestion.load_image(img_path)
            
            parser = None
            if self.xml_paths and idx < len(self.xml_paths):
                parser = CBADParser(self.xml_paths[idx])
                
            # Compute dynamic patch size based on actual XML ground truth!
            if parser and self.patch_size is None:
                current_patch_size = parser.get_optimal_patch_size(zeta=3.0)
            else:
                current_patch_size = self.patch_size or 256
            
            # 2. Extract all patches for this specific image using Context-Adaptive sizing
            patches = self.patch_extractor.extract_patches(image, current_patch_size)
            
            # Shuffle patches so the training batch isn't sequential from the same corner
            random.shuffle(patches)
            
            for p in patches:
                if parser:
                    # Generate Ground Truth Labels on-the-fly to save RAM
                    gt_lines, gt_energy = parser.get_patch_ground_truth(p["bbox"], target_size=self.patch_extractor.target_size)
                    gt_lines_tensor = torch.tensor(gt_lines, dtype=torch.float32) if gt_lines else torch.empty((0, 3))
                    gt_energy_tensor = torch.tensor(gt_energy, dtype=torch.float32).unsqueeze(0)
                    
                    yield p["patch_tensor"], gt_lines_tensor, gt_energy_tensor
                else:
                    yield p["patch_tensor"], p["bbox"], img_path

    @staticmethod
    def custom_collate_fn(batch):
        """
        Custom collate function required because 'gt_lines' has a variable length 
        (different number of lines per patch) so it cannot be stacked uniformly.
        """
        patch_tensors = []
        gt_lines_list = []
        gt_energy_maps = []
        
        for item in batch:
            patch_tensors.append(item[0])
            gt_lines_list.append(item[1])
            gt_energy_maps.append(item[2])
            
        # Stack images and energy maps uniformly
        patch_tensors = torch.stack(patch_tensors, dim=0)
        gt_energy_maps = torch.stack(gt_energy_maps, dim=0)
        
        # Keep gt_lines as a Python list of tensors
        return patch_tensors, gt_lines_list, gt_energy_maps

# --------------------------------------------------------------------------------
# Example usage / Pipeline integration preview (Main logic goes to Phase 6)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy mock function to simulate Phase 2 model estimating line gap
    def mock_model_predict_gap(resized_patch):
        return 15.0 # pixels
        
    print("Initializing Phase 1: Data Preparation & Context-Adaptive Patching")
    
    # Example flow
    # 1. Setup
    sampler = DynamicSampler(target_size=256, zeta=3.0)
    extractor = PatchExtractor(target_size=256, overlap_ratio=0.25)
    
    # 2. Load Image
    # image = DataIngestion.load_image("path/to/historical_manuscript.jpg")
    
    # 3. Compute optimal patch size dynamically
    # t = sampler.compute_optimal_patch_size(image, mock_model_predict_gap)
    # print(f"Computed optimal patch size: {t}")
    
    # 4. Extract patches
    # patches_data = extractor.extract_patches(image, patch_size=t)
    # print(f"Extracted {len(patches_data)} patches from the image.")
