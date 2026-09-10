import numpy as np

class SeamGenerator:
    """
    Handles Phase 5: Stage-2 Seam Generation.
    Uses Dynamic Programming (Seam Carving) to find optimal pathways through the 
    continuous Text-Energy map, constrained by the mathematical scribbles.
    """
    def __init__(self, energy_map, global_scribbles):
        """
        Args:
            energy_map (np.ndarray): 2D float array [0, 1] representing text density.
            global_scribbles (list): Sorted list of global scribbles [gx, gy, gm].
        """
        self.energy = energy_map
        # Ensure scribbles are sorted from top of document to bottom
        self.scribbles = sorted(global_scribbles, key=lambda x: x[1])
        self.h, self.w = energy_map.shape

    def _calculate_cost_matrix(self, y_start_func, y_end_func):
        """
        Calculates the cumulative minimum energy (cost) matrix for a left-to-right seam.
        The search space is dynamically constrained by an upper and lower boundary function.
        """
        # Initialize cost matrix to infinity
        cost = np.full((self.h, self.w), np.inf)
        
        # 1. Initialize the first column based on constraints
        for y in range(self.h):
            if y_start_func(0) <= y <= y_end_func(0):
                cost[y, 0] = self.energy[y, 0]
                
        # 2. Dynamic Programming: Build cumulative energy map left-to-right
        for x in range(1, self.w):
            min_y = max(0, int(y_start_func(x)))
            max_y = min(self.h - 1, int(y_end_func(x)))
            
            # Avoid invalid bounds if lines cross weirdly
            if min_y > max_y:
                continue
                
            for y in range(min_y, max_y + 1):
                # Look at 3 adjacent pixels in the previous column (Up-Left, Left, Down-Left)
                idx_up   = max(0, y - 1)
                idx_mid  = y
                idx_down = min(self.h - 1, y + 1)
                
                min_prev_cost = min(cost[idx_up, x-1], cost[idx_mid, x-1], cost[idx_down, x-1])
                
                # Cumulative cost = Current pixel energy + cheapest path to get here
                cost[y, x] = self.energy[y, x] + min_prev_cost
                
        return cost

    def _backtrack_seam(self, cost_matrix, y_start_func, y_end_func):
        """
        Walks backwards from the right edge to the left edge along the cheapest path 
        to mathematically define the final wavy seam.
        """
        seam = np.zeros(self.w, dtype=int)
        
        end_x = self.w - 1
        min_y = max(0, int(y_start_func(end_x)))
        max_y = min(self.h - 1, int(y_end_func(end_x)))
        
        # Fallback if boundaries collapse
        if min_y > max_y: 
            return seam 
            
        # Find minimum cost point in the last column to start the backtrack
        best_y = min_y + np.argmin(cost_matrix[min_y:max_y+1, end_x])
        seam[end_x] = best_y
        
        # Backtrack leftwards
        for x in range(end_x - 1, -1, -1):
            y = seam[x+1]
            
            # Check the 3 possible predecessors that could have led to pixel 'y'
            idx_up   = max(0, y - 1)
            idx_mid  = y
            idx_down = min(self.h - 1, y + 1)
            
            costs = [cost_matrix[idx_up, x], cost_matrix[idx_mid, x], cost_matrix[idx_down, x]]
            choice = np.argmin(costs)
            
            if choice == 0:
                seam[x] = idx_up
            elif choice == 1:
                seam[x] = idx_mid
            else:
                seam[x] = idx_down
                
        return seam

    def generate_polygons(self):
        """
        Orchestrates the creation of separating seams between every text line, 
        and pairs adjacent seams to form closed text-line polygons.
        
        Returns:
            list: A list of polygons. Each polygon is a list of (x, y) tuples forming the boundary.
        """
        if len(self.scribbles) == 0:
            return []
            
        seams = []
        
        # Helper to calculate mathematical line bounds: y = m(x - mu_x) + mu_y
        def line_eq(scribble, x):
            gx, gy, gm = scribble
            return gy + gm * (x - gx)
            
        # ---------------------------------------------------------
        # Step 1. Generate Topmost seam (above the very first line)
        # ---------------------------------------------------------
        s0 = self.scribbles[0]
        def top_bound(x): return 0
        def bottom_bound(x): return line_eq(s0, x)
        
        cost_mat = self._calculate_cost_matrix(top_bound, bottom_bound)
        seams.append(self._backtrack_seam(cost_mat, top_bound, bottom_bound))
        
        # ---------------------------------------------------------
        # Step 2. Generate Intermediate seams (between consecutive scribbles)
        # ---------------------------------------------------------
        for i in range(len(self.scribbles) - 1):
            s_top = self.scribbles[i]
            s_bot = self.scribbles[i+1]
            
            def top_b(x): return line_eq(s_top, x)
            def bot_b(x): return line_eq(s_bot, x)
            
            cost_mat = self._calculate_cost_matrix(top_b, bot_b)
            seams.append(self._backtrack_seam(cost_mat, top_b, bot_b))
            
        # ---------------------------------------------------------
        # Step 3. Generate Bottommost seam (below the very last line)
        # ---------------------------------------------------------
        s_last = self.scribbles[-1]
        def top_bound_last(x): return line_eq(s_last, x)
        def bottom_bound_last(x): return self.h - 1
        
        cost_mat = self._calculate_cost_matrix(top_bound_last, bottom_bound_last)
        seams.append(self._backtrack_seam(cost_mat, top_bound_last, bottom_bound_last))
        
        # ---------------------------------------------------------
        # Step 4. Form Closed Polygons
        # ---------------------------------------------------------
        # Polygon [i] is formed by walking right along seam[i], then left along seam[i+1]
        polygons = []
        for i in range(len(self.scribbles)):
            upper_seam = seams[i]
            lower_seam = seams[i+1]
            
            # Top edge: (x, y) from x=0 to x=Width
            top_edge = [(x, upper_seam[x]) for x in range(self.w)]
            
            # Bottom edge: (x, y) from x=Width down to x=0 (walking backwards)
            bottom_edge = [(x, lower_seam[x]) for x in range(self.w - 1, -1, -1)]
            
            # Merge to form a closed loop
            polygon = top_edge + bottom_edge
            polygons.append(polygon)
            
        return polygons
