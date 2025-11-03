import torch
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from generator import *
import matplotlib.pyplot as plt
from pytorch_msssim import ssim  

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate(generator, val_loader, vgg_loss, device, epoch=None, save_dir='validation_samples'):
    generator.eval()
    total_g_loss = 0
    total_batches = 0
    
    # Create save directory
    if epoch is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (visual_data, tactile_data) in enumerate(val_loader):
            visual_data = visual_data.to(device)
            real_tactile = tactile_data.to(device)
            
            # Generate fake tactile data
            fake_tactile = generator(visual_data)
            
            # Calculate VGG loss
            g_loss = vgg_loss(fake_tactile, real_tactile)
            
            total_g_loss += g_loss.item()
            total_batches += 1

            # Save validation samples
            if epoch is not None and i == 0:
                total_samples = real_tactile.size(0)
                num_saves = 16
                
                interval = max(1, total_samples // num_saves)
                save_indices = range(0, total_samples, interval)[:num_saves]
                
                real_samples = real_tactile[save_indices].cpu()
                fake_samples = fake_tactile[save_indices].cpu()
                
                fig = plt.figure(figsize=(15, 5 * num_saves))
                
                for img_idx in range(num_saves):
                    real_img = real_samples[img_idx]
                    fake_img = fake_samples[img_idx]
                    
                    for j in range(3):
                        plt.subplot(num_saves, 6, img_idx * 6 + j + 1)
                        plt.imshow(real_img[j], cmap='gray')
                        plt.axis('off')
                        plt.title(f'Real Ch{j+1}' if img_idx == 0 else '')
                        
                        plt.subplot(num_saves, 6, img_idx * 6 + j + 4)
                        plt.imshow(fake_img[j], cmap='gray')
                        plt.axis('off')
                        plt.title(f'Fake Ch{j+1}' if img_idx == 0 else '')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'validation_epoch_{epoch}.png'))
                plt.close()
                break
    
    generator.train()
    return total_g_loss / total_batches

def print_model_size(model):  
    total_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    
    # 计算MB大小 (假设每个参数是float32，占4字节)  
    total_mb = total_params * 4 / (1024 * 1024)  
    trainable_mb = trainable_params * 4 / (1024 * 1024)  
    
    print(f'Generator Total Parameters: {total_params:,} ({total_mb:.2f} MB)')  
    print(f'Generator Trainable Parameters: {trainable_params:,} ({trainable_mb:.2f} MB)')  

def train(  
    train_loader,  
    val_loader,  
    generator,  
    optimizer,  
    vgg_loss,  
    device,  
    num_epochs=100,  
    save_interval=10,  
    checkpoint_dir='ckpts',  
    log_dir='runs',  
    resume_checkpoint=None  
):  
    os.makedirs(checkpoint_dir, exist_ok=True)  
    torch.autograd.set_detect_anomaly(True)  

    start_epoch = 0  
    best_ssim = -float('inf')  # Changed from best_val_loss to best_ssim  
    if resume_checkpoint:  
        checkpoint = torch.load(resume_checkpoint, map_location=device)  
        generator.load_state_dict(checkpoint['generator_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        start_epoch = checkpoint['epoch'] + 1  
        # best_ssim = checkpoint.get('best_ssim', -float('inf'))  # Load best_ssim if available  
        log_dir = os.path.join(log_dir, f'resume_from_epoch_{start_epoch}')  

    writer = SummaryWriter(log_dir)  

    print_model_size(generator)  
    generator.train()  
    print(len(train_loader.dataset), len(val_loader.dataset))  
    print(f"Starting training from epoch {start_epoch} to {num_epochs}")  
    
    for epoch in range(start_epoch, num_epochs):  
        # Training phase  
        total_g_loss = 0  
        total_batches = 0  
        
        for i, (visual_data, tactile_data) in enumerate(train_loader):  
            visual_data = visual_data.to(device)  
            real_tactile = tactile_data.to(device)  
            
            # Train Generator  
            optimizer.zero_grad()  
            
            fake_tactile = generator(visual_data)  
            g_loss = vgg_loss(fake_tactile, real_tactile)  
            
            g_loss.backward()  
            optimizer.step()  
            
            total_g_loss += g_loss.item()  
            total_batches += 1  
            
            if i % 100 == 0:  
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '  
                      f'Loss: {g_loss.item():.4f}')  
        
        # Calculate average training loss  
        avg_train_loss = total_g_loss / total_batches  
        
        # Validation phase with SSIM calculation  
        generator.eval()  
        total_ssim = 0  
        val_batches = 0  
        
        with torch.no_grad():  
            idx_val = 0
            for visual_data, tactile_data in val_loader:  
                if idx_val > 100:
                    break
                idx_val += 1
                visual_data = visual_data.to(device)  
                real_tactile = tactile_data.to(device)  
                
                fake_tactile = generator(visual_data)  
                batch_ssim = ssim(fake_tactile, real_tactile, data_range=1.0)  
                
                total_ssim += batch_ssim.item()  
                val_batches += 1  
        
        avg_ssim = total_ssim / val_batches  
        generator.train()  

        # save validation images
        validate(
            generator,
            val_loader,
            vgg_loss,
            device,
            epoch=epoch,
            save_dir=os.path.join(log_dir, 'validation_samples')
        )

        # Log to tensorboard  
        writer.add_scalar('Loss/train', avg_train_loss, epoch)  
        writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)  
        
        # Save best model based on SSIM  
        if avg_ssim > best_ssim:  
            best_ssim = avg_ssim  
            print(f"New best SSIM: {best_ssim:.4f}! Saving checkpoint...")  
            torch.save({  
                'generator_state_dict': generator.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'epoch': epoch,  
                'best_ssim': best_ssim,  
            }, os.path.join(checkpoint_dir, 'best_model.pth'))  
        
        # Regular interval checkpoints  
        if epoch % save_interval == 0:  
            torch.save({  
                'generator_state_dict': generator.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'epoch': epoch,  
                'best_ssim': best_ssim,  
            }, os.path.join(checkpoint_dir, f'epoch_{epoch}.pth'))  

        print(f'Epoch [{epoch}/{num_epochs}], '  
              f'Train Loss: {avg_train_loss:.4f}, '  
              f'SSIM: {avg_ssim:.4f}')  
    
    # Save final model  
    torch.save({  
        'generator_state_dict': generator.state_dict(),  
        'optimizer_state_dict': optimizer.state_dict(),  
        'epoch': num_epochs-1,  
        'best_ssim': best_ssim,  
    }, os.path.join(checkpoint_dir, 'final_model.pth'))  
    print("Final model saved successfully!")  

    writer.close()  

def main():
    # Set random seed
    seed = 42
    set_seed(seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    base_dir = 'data/20251103_113812'
    train_loader, val_loader, test_loader = get_dataloaders(
        base_dir=base_dir,
        batch_size=128,
        num_workers=64,
        seed=seed
    )
    
    generator = ResnetGenerator2(
        input_shape=(9, 128, 128),
        output_channels=1,
        dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(  
        generator.parameters(),  
        lr=3e-4,
        betas=(0.5, 0.999),  
        eps=1e-8  
    )  
    
    # Initialize VGG loss
    vgg_loss = VGGLoss().to(device)
    
    # Create directories for checkpoints and logs
    checkpoint_dir = 'ckpts'
    log_dir = 'runs'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Start training
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        generator=generator,
        optimizer=optimizer,
        vgg_loss=vgg_loss,
        device=device,
        num_epochs=1000,
        save_interval=50,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        resume_checkpoint=None
    )

if __name__ == '__main__':
    main()