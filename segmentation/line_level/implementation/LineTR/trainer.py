import os
import torch
import torch.optim as optim
from tqdm import tqdm

class LineTRTrainer:
    """
    Orchestrates the specialized Two-Phase training strategy for LineTR.
    """
    def __init__(self, model, line_loss_fn, energy_loss_fn, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.line_loss_fn = line_loss_fn
        self.energy_loss_fn = energy_loss_fn
        
    def _set_requires_grad(self, nets, requires_grad=False):
        """Utility to freeze/unfreeze network components."""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_phase_1(self, dataloader, epochs_part1=60, epochs_part2=20, checkpoint_dir="checkpoints", resume_from=None, eval_loader=None):
        """
        Phase 1: Train ONLY the ViT Backbone and Line-Parameter Generator.
        The Energy Generator remains completely frozen.
        """
        print(">>> Starting Training Phase 1 (Backbone + Line Generator)")
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        # 1. Freeze Energy Generator
        self._set_requires_grad(self.model.energy_generator, False)
        # 2. Unfreeze Backbone and Line Generator
        self._set_requires_grad([self.model.backbone, self.model.line_generator], True)
        
        optimizer = optim.AdamW([
            {'params': self.model.backbone.parameters()},
            {'params': self.model.line_generator.parameters()}
        ], lr=5e-5)
        
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"[*] Resuming Phase 1 from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[*] Resuming at epoch {start_epoch + 1}")
        
        total_epochs = epochs_part1 + epochs_part2
        
        for epoch in range(start_epoch, total_epochs):
            if epoch == epochs_part1:
                print(">> Reducing learning rate to 1e-5")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
                    
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Phase 1 Epoch [{epoch+1}/{total_epochs}]")
            for batch_idx, (images, gt_lines, _) in enumerate(pbar):
                images = images.to(self.device)
                optimizer.zero_grad()
                
                pred_lines, pred_probs, _ = self.model(images)
                gt_lines_dev = [g.to(self.device) for g in gt_lines]
                loss = self.line_loss_fn(pred_lines, pred_probs, gt_lines_dev)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Phase 1 - Epoch [{epoch+1}/{total_epochs}] Completed. Average Loss: {avg_loss:.4f}")
            
            # --- Validation Loop ---
            val_loss = float('inf')
            if eval_loader:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    val_pbar = tqdm(eval_loader, desc=f"Val Phase 1 Epoch [{epoch+1}/{total_epochs}]")
                    for batch_idx, (images, gt_lines, _) in enumerate(val_pbar):
                        images = images.to(self.device)
                        pred_lines, pred_probs, _ = self.model(images)
                        gt_lines_dev = [g.to(self.device) for g in gt_lines]
                        
                        loss = self.line_loss_fn(pred_lines, pred_probs, gt_lines_dev)
                        val_loss_sum += loss.item()
                        val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                        
                val_loss = val_loss_sum / len(eval_loader)
                print(f"Phase 1 - Epoch [{epoch+1}/{total_epochs}] Validation Loss: {val_loss:.4f}")
            
            # --- Checkpointing ---
            if checkpoint_dir:
                ckpt_path = os.path.join(checkpoint_dir, f"linetr_phase1_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss
                }, ckpt_path)
                
                # Save the "Best" model to prevent overfitting
                if not hasattr(self, 'best_phase1_val_loss'):
                    self.best_phase1_val_loss = float('inf')
                    
                if val_loss < self.best_phase1_val_loss:
                    self.best_phase1_val_loss = val_loss
                    best_ckpt_path = os.path.join(checkpoint_dir, "linetr_phase1_best.pth")
                    torch.save(self.model.state_dict(), best_ckpt_path)
                    print(f"[*] New Best Phase 1 Model Saved! (Val Loss: {val_loss:.4f})")
                
        # Save final clean weights
        if checkpoint_dir:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "linetr_phase1_final.pth"))

    def train_phase_2(self, dataloader, epochs=50, checkpoint_dir="checkpoints", resume_from=None, eval_loader=None):
        """
        Phase 2: Train ONLY the Text-Energy Map Generator.
        The Backbone and Line Generator are locked as foundational feature extractors.
        """
        print(">>> Starting Training Phase 2 (Text-Energy Map Generator)")
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        # 1. Freeze Backbone and Line Generator
        self._set_requires_grad([self.model.backbone, self.model.line_generator], False)
        # 2. Unfreeze Energy Generator
        self._set_requires_grad(self.model.energy_generator, True)
        
        optimizer = optim.AdamW(self.model.energy_generator.parameters(), lr=1e-4)
        
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"[*] Resuming Phase 2 from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[*] Resuming at epoch {start_epoch + 1}")
            
        for epoch in range(start_epoch, epochs):
            self.model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Phase 2 Epoch [{epoch+1}/{epochs}]")
            for batch_idx, (images, _, gt_energy_maps) in enumerate(pbar):
                images = images.to(self.device)
                gt_energy_maps = gt_energy_maps.to(self.device)
                
                optimizer.zero_grad()
                
                _, _, pred_energy_maps = self.model(images)
                loss = self.energy_loss_fn(pred_energy_maps, gt_energy_maps)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            avg_loss = epoch_loss / len(dataloader)
            print(f"Phase 2 - Epoch [{epoch+1}/{epochs}] Completed. Average Loss: {avg_loss:.4f}")
            
            # --- Validation Loop ---
            val_loss = float('inf')
            if eval_loader:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    val_pbar = tqdm(eval_loader, desc=f"Val Phase 2 Epoch [{epoch+1}/{epochs}]")
                    for batch_idx, (images, _, gt_energy_maps) in enumerate(val_pbar):
                        images = images.to(self.device)
                        gt_energy_maps = gt_energy_maps.to(self.device)
                        
                        _, _, pred_energy_maps = self.model(images)
                        loss = self.energy_loss_fn(pred_energy_maps, gt_energy_maps)
                        
                        val_loss_sum += loss.item()
                        val_pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                        
                val_loss = val_loss_sum / len(eval_loader)
                print(f"Phase 2 - Epoch [{epoch+1}/{epochs}] Validation Loss: {val_loss:.4f}")
            
            if checkpoint_dir:
                ckpt_path = os.path.join(checkpoint_dir, f"linetr_phase2_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'val_loss': val_loss
                }, ckpt_path)
                
                # Best Model Checkpointing
                if not hasattr(self, 'best_phase2_val_loss'):
                    self.best_phase2_val_loss = float('inf')
                    
                if val_loss < self.best_phase2_val_loss:
                    self.best_phase2_val_loss = val_loss
                    best_ckpt_path = os.path.join(checkpoint_dir, "linetr_phase2_best.pth")
                    torch.save(self.model.state_dict(), best_ckpt_path)
                    print(f"[*] New Best Phase 2 Model Saved! (Val Loss: {val_loss:.4f})")
                
        if checkpoint_dir:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "linetr_final.pth"))
