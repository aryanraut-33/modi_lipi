import numpy as np
import cv2

class PostProcessor:
    """
    Handles Phase 4: Taking local patch-level predictions and stitching them back 
    into the full-resolution global document coordinate space.
    """
    def __init__(self, original_image_shape, model_target_size=256):
        """
        Args:
            original_image_shape (tuple): (Height, Width, Channels) of the full document.
            model_target_size (int): The 256x256 size that the network outputted.
        """
        self.doc_h, self.doc_w, _ = original_image_shape
        self.target = model_target_size
        
        # Canvas for stitching the continuous energy map
        self.global_energy = np.zeros((self.doc_h, self.doc_w), dtype=np.float32)
        self.overlap_counts = np.zeros((self.doc_h, self.doc_w), dtype=np.float32)
        
        # Master list to collect all un-normalized scribbles
        self.global_scribbles = []

    def unnormalize_scribbles(self, patch_scribbles, patch_probs, bbox, prob_threshold=0.9):
        """
        Reverses the coordinate normalization. Maps the neural network's [0, 1] 
        predictions back to exact pixel coordinates on the giant document.
        
        Args:
            patch_scribbles: (N, 3) -> [mu_x, mu_y, m] for this patch.
            patch_probs: (N) confidence scores for this patch.
            bbox: (x1, y1, x2, y2) original extraction coordinates in the global document.
        """
        x1, y1, x2, y2 = bbox
        patch_w = x2 - x1
        patch_h = y2 - y1
        
        # Convert tensors to numpy if necessary
        if hasattr(patch_scribbles, "cpu"):
            patch_scribbles = patch_scribbles.detach().cpu().numpy()
            patch_probs = patch_probs.detach().cpu().numpy()
        
        valid_scribbles = []
        for i in range(len(patch_probs)):
            # Only keep highly confident scribbles
            if patch_probs[i] > prob_threshold:
                mu_x, mu_y, m = patch_scribbles[i]
                
                # Convert normalized [0,1] patch coords to absolute global coords
                global_x = x1 + (mu_x * patch_w)
                global_y = y1 + (mu_y * patch_h)
                
                # Because we resized square patches uniformly (t x t -> 256 x 256),
                # the slope (dy/dx) remains mathematically identical. No scaling needed for 'm'.
                global_m = m
                
                valid_scribbles.append([global_x, global_y, global_m])
                
        self.global_scribbles.extend(valid_scribbles)

    def stitch_energy_map(self, patch_energy, bbox):
        """
        Blends a single 256x256 patch energy map onto the global document canvas.
        
        Args:
            patch_energy: (256, 256) continuous map [0, 1] for this patch.
            bbox: (x1, y1, x2, y2) original extraction coordinates.
        """
        x1, y1, x2, y2 = bbox
        patch_w = x2 - x1
        patch_h = y2 - y1
        
        if hasattr(patch_energy, "cpu"):
            patch_energy = patch_energy.detach().cpu().numpy()
            
        # Ensure it's 2D
        if patch_energy.ndim > 2:
            patch_energy = np.squeeze(patch_energy)
            
        # Un-resize the 256x256 energy map back up to its original extraction size (t x t)
        patch_energy_resized = cv2.resize(patch_energy, (patch_w, patch_h))
        
        # Add to the overlapping global canvas
        self.global_energy[y1:y2, x1:x2] += patch_energy_resized
        self.overlap_counts[y1:y2, x1:x2] += 1.0

    def finalize_energy_map(self):
        """
        Averages out the overlapping regions to create a seamless, smooth global energy map.
        
        Returns:
            np.ndarray: The finalized global continuous [0, 1] energy map.
        """
        # Avoid division by zero in blank spots
        safe_counts = np.maximum(self.overlap_counts, 1.0)
        final_map = self.global_energy / safe_counts
        
        # Ensure it stays bounded in [0, 1]
        final_map = np.clip(final_map, 0.0, 1.0)
        return final_map

    def iterative_projection_merging(self, distance_threshold=20.0, slope_threshold=0.2):
        """
        Clustering Algorithm. Groups hundreds of overlapping, nearby patch-level 
        scribbles into unified, document-level text-line scribbles.
        
        Returns:
            list: The merged global scribbles [global_x, global_y, global_m].
        """
        if not self.global_scribbles:
            return []
            
        scribbles = np.array(self.global_scribbles)
        
        # Sort all scribbles top-to-bottom based on their Y coordinate
        scribbles = scribbles[scribbles[:, 1].argsort()]
        
        clusters = []
        
        for scribble in scribbles:
            gx, gy, gm = scribble
            
            # Try to match the scribble to an existing cluster
            matched = False
            for cluster in clusters:
                # A cluster is represented by the average of its contents
                cx, cy, cm = np.mean(cluster, axis=0)
                
                # Check spatial vertical distance and slope similarity
                if abs(gy - cy) < distance_threshold and abs(gm - cm) < slope_threshold:
                    cluster.append(scribble)
                    matched = True
                    break
                    
            if not matched:
                # If it doesn't fit anywhere, start a new cluster
                clusters.append([scribble])
                
        # Merge clusters by averaging their parameters
        merged_global_lines = []
        for cluster in clusters:
            avg_x, avg_y, avg_m = np.mean(cluster, axis=0)
            merged_global_lines.append([avg_x, avg_y, avg_m])
            
        return merged_global_lines
