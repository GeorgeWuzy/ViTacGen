import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

class TactileGANDataset(Dataset):  
    def __init__(self, episode_dirs, transform=None):  
        self.transform = transform  
        self.samples = []  
        
        print(f"Initializing dataset with {len(episode_dirs)} episode directories")  
        
        # Collect all valid sample sequences  
        for episode_dir in episode_dirs:  
            visual_dir = os.path.join(episode_dir, 'visual')  
            tactile_dir = os.path.join(episode_dir, 'tactile')  
            
            if not os.path.exists(visual_dir) or not os.path.exists(tactile_dir):  
                print(f"Warning: Directory not found - Visual: {visual_dir}, Tactile: {tactile_dir}")  
                continue  
            
            # Get sorted frame paths  
            visual_frames = sorted(glob(os.path.join(visual_dir, '*.png')))  
            tactile_frames = sorted(glob(os.path.join(tactile_dir, '*.png')))  
            
            if len(visual_frames) == 0 or len(tactile_frames) == 0:  
                print(f"Warning: No frames found in {episode_dir}")  
                print(f"Visual frames: {len(visual_frames)}, Tactile frames: {len(tactile_frames)}")  
                continue  
                
            if len(visual_frames) != len(tactile_frames):  
                print(f"Warning: Mismatched frame counts in {episode_dir}")  
                print(f"Visual frames: {len(visual_frames)}, Tactile frames: {len(tactile_frames)}")  
                continue  
            
            # Create non-overlapping sequences of 3 frames  
            for i in range(0, len(visual_frames) - 2, 3):  
                self.samples.append({  
                    'visual': visual_frames[i:i+3],  
                    'tactile': tactile_frames[i:i+3]  
                })  
        
        print(f"Total valid samples collected: {len(self.samples)}")  

    def __getitem__(self, idx):  
        sample = self.samples[idx]  
        
        # Load and process visual frames  
        visual_frames = []  
        for v_path in sample['visual']:  
            if not os.path.exists(v_path):  
                raise FileNotFoundError(f"Visual frame not found: {v_path}")  
                
            # Read as RGB  
            frame = cv2.imread(v_path)  
            if frame is None:  
                raise ValueError(f"Failed to load visual frame: {v_path}")  
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = cv2.resize(frame, (128, 128))  
            frame = frame / 255.0  
            visual_frames.append(frame)  
        
        # Load and process tactile frames  
        tactile_frames = []  
        for t_path in sample['tactile']:  
            if not os.path.exists(t_path):  
                raise FileNotFoundError(f"Tactile frame not found: {t_path}")  
                
            # Read as grayscale  
            frame = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)  
            if frame is None:  
                raise ValueError(f"Failed to load tactile frame: {t_path}")  
                
            frame = cv2.resize(frame, (128, 128))  
            frame = frame / 255.0  
            tactile_frames.append(frame)  
        
        try:  
            # Concatenate frames  
            visual_data = np.concatenate(visual_frames, axis=2)  
            tactile_data = np.stack(tactile_frames, axis=0)  
            
            # Convert to torch tensors  
            visual_data = torch.FloatTensor(visual_data.transpose(2, 0, 1))  
            tactile_data = torch.FloatTensor(tactile_data)  
            
            return visual_data, tactile_data  
            
        except Exception as e:  
            print(f"Error processing sample {idx}: {str(e)}")  
            print(f"Visual paths: {sample['visual']}")  
            print(f"Tactile paths: {sample['tactile']}")  
            raise  
    
    def __len__(self):  
        return len(self.samples)  
        

def get_dataloaders(base_dir, batch_size=32, num_workers=4, seed=42):  
    """  
    Create train, validation, and test dataloaders  
    """  
    if not os.path.exists(base_dir):  
        raise FileNotFoundError(f"Base directory not found: {base_dir}")  
    
    # Get all episode directories  
    episode_dirs = sorted(glob(os.path.join(base_dir, 'episode_*')))  
    
    if len(episode_dirs) == 0:  
        raise ValueError(f"No episode directories found in {base_dir}")  
    
    print(f"Found {len(episode_dirs)} episode directories")  
    
    # Split episodes  
    train_episodes, temp_episodes = train_test_split(episode_dirs, test_size=0.3, random_state=seed)  
    val_episodes, test_episodes = train_test_split(temp_episodes, test_size=0.33, random_state=seed)  
    
    print(f"Split sizes - Train: {len(train_episodes)}, Val: {len(val_episodes)}, Test: {len(test_episodes)}")  
    
    # Create datasets  
    train_dataset = TactileGANDataset(train_episodes)  
    val_dataset = TactileGANDataset(val_episodes)  
    test_dataset = TactileGANDataset(test_episodes)  
    
    # Create dataloaders with error handling  
    try:  
        train_loader = DataLoader(  
            train_dataset,  
            batch_size=batch_size,  
            shuffle=True,  
            num_workers=num_workers,  
            pin_memory=True  
        )  
        
        val_loader = DataLoader(  
            val_dataset,  
            batch_size=batch_size,  
            shuffle=False,  
            num_workers=num_workers,  
            pin_memory=True  
        )  
        
        test_loader = DataLoader(  
            test_dataset,  
            batch_size=batch_size,  
            shuffle=False,  
            num_workers=num_workers,  
            pin_memory=True  
        )  
        
        return train_loader, val_loader, test_loader  
        
    except Exception as e:  
        print(f"Error creating dataloaders: {str(e)}")  
        raise