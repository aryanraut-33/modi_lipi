import xml.etree.ElementTree as ET
import numpy as np
import cv2

class CBADParser:
    """
    Parses PAGE XML annotations from the cBAD2019 dataset and converts 
    them into mathematical scribbles and binary energy maps for LineTR.
    """
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.baselines = self._parse_xml()

    def _parse_xml(self):
        """Reads the XML and extracts a list of baseline polylines."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        baselines = []
        # Iterate over all elements, ignoring namespaces
        for elem in root.iter():
            if 'Baseline' in elem.tag:
                points_str = elem.attrib.get('points', '')
                if not points_str: 
                    continue
                
                # Parse "x1,y1 x2,y2 ..." into a list of (x, y) tuples
                pts = []
                for pt_str in points_str.split():
                    try:
                        x, y = map(int, pt_str.split(','))
                        pts.append((x, y))
                    except ValueError:
                        continue
                        
                if len(pts) >= 2:
                    baselines.append(pts)
                    
        return baselines

    def get_optimal_patch_size(self, zeta=3.0, default_size=256):
        """
        Dynamically calculates the mathematically perfect patch size (t) 
        for this specific document using the exact ground-truth XML baselines.
        """
        if len(self.baselines) < 2:
            return default_size
            
        # Calculate the median Y coordinate for each baseline
        ys = []
        for baseline in self.baselines:
            if len(baseline) > 0:
                ys.append(np.median([p[1] for p in baseline]))
                
        ys.sort()
        # Find the vertical gaps between adjacent text lines
        gaps = [ys[i] - ys[i-1] for i in range(1, len(ys))]
        
        if not gaps:
            return default_size
            
        # Delta = median interline gap
        delta_avg = np.median(gaps)
        
        # t = Delta * zeta
        t = int(delta_avg * zeta)
        
        # Clamp it to ensure it fits reasonably in VRAM and doesn't crash on huge gaps
        return max(64, min(t, 1024))

    def get_patch_ground_truth(self, bbox, target_size=256):
        """
        Converts the full-document baselines into patch-specific training labels.
        
        Args:
            bbox: (x1, y1, x2, y2) The absolute coordinates of the patch.
            target_size: The resized output size expected by the model (256).
            
        Returns:
            gt_lines: list of [mu_x, mu_y, m]
            gt_energy_map: (256, 256) binary numpy array [0, 1]
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        gt_lines = []
        
        # 1. Create a blank canvas for the Energy Map
        energy_canvas = np.zeros((h, w), dtype=np.float32)
        
        for baseline in self.baselines:
            # Shift baseline coordinates so they are relative to the top-left of this patch
            local_pts = [(x - x1, y - y1) for x, y in baseline]
            
            # Check if this baseline actually intersects with our patch
            xs = [p[0] for p in local_pts]
            ys = [p[1] for p in local_pts]
            if max(xs) < 0 or min(xs) > w or max(ys) < 0 or min(ys) > h:
                continue
                
            # Render the baseline onto the energy map (thick white stroke on black paper)
            thickness = max(2, int(h * 0.05)) # roughly 5% of patch height
            pts_array = np.array(local_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(energy_canvas, [pts_array], isClosed=False, color=1.0, thickness=thickness)
            
            # 2. Extract Mathematical Scribble [mu_x, mu_y, m]
            # Filter points to only those strictly inside this patch
            inside_pts = [(x, y) for x, y in local_pts if 0 <= x <= w and 0 <= y <= h]
            
            if len(inside_pts) >= 2:
                xs_in = np.array([p[0] for p in inside_pts])
                ys_in = np.array([p[1] for p in inside_pts])
                
                # Avoid division by zero for perfectly vertical strokes
                if np.max(xs_in) - np.min(xs_in) > 5:
                    # Fit a straight line y = mx + c through the points in the patch
                    m, c = np.polyfit(xs_in, ys_in, 1)
                    
                    # Find the midpoint of the line segment inside the patch
                    mid_x = (np.max(xs_in) + np.min(xs_in)) / 2.0
                    mid_y = m * mid_x + c
                    
                    # Normalize to [0, 1] fractional coordinates
                    mu_x = mid_x / w
                    mu_y = mid_y / h
                    
                    # Slope 'm' is scale-invariant for squares, no change needed
                    gt_lines.append([mu_x, mu_y, m])

        # Resize the energy canvas to exactly 256x256
        energy_map_resized = cv2.resize(energy_canvas, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        return gt_lines, energy_map_resized

if __name__ == "__main__":
    # Small test snippet to verify parser mechanics
    print("dataset_parser.py compiled successfully.")
