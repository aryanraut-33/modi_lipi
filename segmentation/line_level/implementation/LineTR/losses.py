import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class LineTRLoss(nn.Module):
    """
    Computes the Bipartite Matching and Optimization Loss for the predicted scribbles.
    """
    def __init__(self, lambda_geom=1.0, lambda_p=1.0, lambda_1=1.0, lambda_2=1.0, focal_gamma=2.0):
        super().__init__()
        self.lambda_geom = lambda_geom
        self.lambda_p = lambda_p
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gamma = focal_gamma

    def focal_loss(self, p, is_positive):
        """
        Calculates standard focal loss.
        """
        eps = 1e-6
        if is_positive:
            return - ((1 - p) ** self.gamma) * torch.log(p + eps)
        else:
            return - (p ** self.gamma) * torch.log(1 - p + eps)

    def compute_geometric_loss(self, pred_line, gt_line):
        """
        Calculates geometric loss based on the vertical distance of the line 
        at the left (x=0) and right (x=1) boundaries of the patch.
        line parameters are [mu_x, mu_y, m]
        Equation: y = mu_y + m * (x - mu_x)
        """
        # Pred bounds
        pred_y_left = pred_line[1] - pred_line[2] * pred_line[0]
        pred_y_right = pred_line[1] + pred_line[2] * (1 - pred_line[0])
        
        # GT bounds
        gt_y_left = gt_line[1] - gt_line[2] * gt_line[0]
        gt_y_right = gt_line[1] + gt_line[2] * (1 - gt_line[0])
        
        d_left = pred_y_left - gt_y_left
        d_right = pred_y_right - gt_y_right
        
        return d_left**2 + d_right**2

    def forward(self, pred_lines, pred_probs, gt_lines_batch):
        """
        Args:
            pred_lines: Tensor of shape (B, N, 3)
            pred_probs: Tensor of shape (B, N, 1)
            gt_lines_batch: List of length B. Each item is a Tensor (M_b, 3)
        """
        B = pred_lines.shape[0]
        total_loss = 0.0
        
        for b in range(B):
            preds = pred_lines[b]           # (N, 3)
            probs = pred_probs[b].squeeze() # (N)
            gts = gt_lines_batch[b]         # (M, 3)
            
            M = gts.shape[0]
            N = preds.shape[0]
            
            # Edge case: No text lines in this patch
            if M == 0:
                loss = self.lambda_1 * sum([self.focal_loss(probs[i], False) for i in range(N)])
                total_loss += loss
                continue
                
            # Calculate delta (median vertical gap between ground truth lines)
            if M > 1:
                sorted_mu_y, _ = torch.sort(gts[:, 1])
                gaps = sorted_mu_y[1:] - sorted_mu_y[:-1]
                delta = torch.median(gaps).item()
                if delta == 0: 
                    delta = 0.01 # Prevent division by zero
            else:
                delta = 0.1 # Fallback if only 1 line exists
                
            # Build Bipartite Cost Matrix
            cost_matrix = torch.zeros(N, M)
            geom_loss_matrix = torch.zeros(N, M)
            
            for i in range(N):
                for j in range(M):
                    l_geom = self.compute_geometric_loss(preds[i], gts[j])
                    geom_loss_matrix[i, j] = l_geom
                    # Cost function explicitly defined in LineTR architecture
                    cost_matrix[i, j] = (self.lambda_geom * l_geom / delta) - (self.lambda_p * probs[i])
                    
            # Hungarian algorithm for optimal matching
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
            matched_indices = set(row_ind)
            
            loss_b = 0.0
            
            # Accumulate loss for un-matched queries (they should predict empty)
            for i in range(N):
                if i not in matched_indices:
                    loss_b += self.lambda_1 * self.focal_loss(probs[i], False)
            
            # Accumulate loss for matched queries
            for idx, i in enumerate(row_ind):
                j = col_ind[idx]
                l_geom = geom_loss_matrix[i, j]
                l_focal = self.lambda_2 * self.focal_loss(probs[i], True)
                
                # Final loss formulation
                loss_b += l_focal + self.lambda_geom * (l_geom / (delta**2))
                
            total_loss += loss_b
            
        return total_loss / B


class TextEnergyLoss(nn.Module):
    """
    Computes loss for the continuous Text-Energy Map Generator using Focal Loss.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_map, gt_map):
        """
        Args:
            pred_map: (B, 1, 256, 256) continuous [0, 1] output from Sigmoid
            gt_map: (B, 1, 256, 256) binary {0, 1} ground truth
        """
        eps = 1e-6
        # Focal Loss formulation for continuous maps
        pos_term = -self.alpha * gt_map * ((1 - pred_map)**self.gamma) * torch.log(pred_map + eps)
        neg_term = -(1 - self.alpha) * (1 - gt_map) * (pred_map**self.gamma) * torch.log(1 - pred_map + eps)
        
        loss = pos_term + neg_term
        return loss.mean()
