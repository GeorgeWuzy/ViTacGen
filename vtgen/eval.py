import torch  
import numpy as np  
import random  
import os  
import time  
from torch.utils.data import DataLoader, Subset  
from torchvision.utils import save_image  
from pytorch_msssim import ssim, ms_ssim  
import lpips  
from dataset import * 
from generator import *  
import matplotlib.pyplot as plt  
from tqdm import tqdm  

def set_seed(seed):  
    """Set random seeds for reproducibility"""  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

def calculate_psnr(img1, img2):  
    """Calculate PSNR between two images"""  
    mse = torch.mean((img1 - img2) ** 2)  
    if mse == 0:  
        return float('inf')  
    max_pixel = 1.0  
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))  
    return psnr  

def print_model_size(model):  
    """Calculate and print model size"""  
    total_params = sum(p.numel() for p in model.parameters())  
    total_size = total_params * 4 / (1024 * 1024) # 4 bytes per parameter  
    print(f'Model Size: {total_size:.2f} MiB')  
    print(f'Total Parameters: {total_params:,}')  
    return total_size  

def measure_inference_speed(model, input_tensor, device, num_iterations=100):  
    """Measure model inference speed"""  
    model.eval()  
    
    # Warmup  
    for _ in range(10):  
        with torch.no_grad():  
            _ = model(input_tensor)  
    
    # Measure time  
    torch.cuda.synchronize()  
    start_time = time.time()  
    
    for _ in range(num_iterations):  
        with torch.no_grad():  
            _ = model(input_tensor)  
    
    torch.cuda.synchronize()  
    end_time = time.time()  
    
    elapsed_time = end_time - start_time  
    fps = num_iterations / elapsed_time  
    
    print(f'Inference Speed: {fps:.2f} FPS')  
    print(f'Average time per image: {(elapsed_time/num_iterations)*1000:.2f} ms')  
    return fps  

def evaluate(generator, eval_loader, device, save_dir):  
    """Evaluate model and calculate metrics"""  
    generator.eval()  
    
    # Initialize metrics  
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  
    total_psnr = 0  
    total_ssim = 0  
    total_lpips = 0  
    count = 0  
    
    # Create directories for saving results  
    os.makedirs(save_dir, exist_ok=True)  
    os.makedirs(os.path.join(save_dir, 'real'), exist_ok=True)  
    os.makedirs(os.path.join(save_dir, 'fake'), exist_ok=True)  
    os.makedirs(os.path.join(save_dir, 'visual'), exist_ok=True)  
    
    results = []  
    
    with torch.no_grad():  
        for i, (visual_data, tactile_data) in enumerate(tqdm(eval_loader)):  
            visual_data = visual_data.to(device)  
            real_tactile = tactile_data.to(device)  
            
            # Generate fake tactile data  
            fake_tactile = generator(visual_data)  
            
            # Calculate metrics  
            psnr = calculate_psnr(real_tactile, fake_tactile)  
            ssim_value = ssim(real_tactile, fake_tactile, data_range=1.0)  
            lpips_value = loss_fn_alex(real_tactile, fake_tactile).mean()  
            
            total_psnr += psnr.item()  
            total_ssim += ssim_value.item()  
            total_lpips += lpips_value.item()  
            count += 1  
            
            # Save results  
            for b in range(real_tactile.size(0)):  
                # save visual data
                for img_idx in range(3):
                    rgb_img = visual_data[b, img_idx*3:(img_idx+1)*3]  
                    save_image(rgb_img,   
                             os.path.join(save_dir, 'visual', f'sample_{i}_{b}_img_{img_idx}.png'))  
                
                # save real tactile
                for c in range(real_tactile.size(1)):  
                    save_image(real_tactile[b, c:c+1],  
                             os.path.join(save_dir, 'real', f'sample_{i}_{b}_channel_{c}.png'))  
                
                # save fake tactile  
                for c in range(fake_tactile.size(1)):  
                    save_image(fake_tactile[b, c:c+1],  
                             os.path.join(save_dir, 'fake', f'sample_{i}_{b}_channel_{c}.png'))  
            
            # Store individual results  
            results.append({  
                'psnr': psnr.item(),  
                'ssim': ssim_value.item(),  
                'lpips': lpips_value.item()  
            })  
    
    # Calculate averages  
    avg_psnr = total_psnr / count  
    avg_ssim = total_ssim / count  
    avg_lpips = total_lpips / count  
    
    # Calculate standard deviations  
    psnr_std = np.std([r['psnr'] for r in results])  
    ssim_std = np.std([r['ssim'] for r in results])  
    lpips_std = np.std([r['lpips'] for r in results])  
    
    metrics = {  
        'PSNR': f"{avg_psnr:.2f} ± {psnr_std:.2f}",  
        'SSIM': f"{avg_ssim:.4f} ± {ssim_std:.4f}",  
        'LPIPS': f"{avg_lpips:.4f} ± {lpips_std:.4f}"  
    }  
    
    return metrics

def main():  
    # Set random seed  
    seed = 42  
    set_seed(seed)  
    
    # Set device  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    # Load the full dataset  
    base_dir = '../collected_data/'  
    _, _, test_loader = get_dataloaders(  
        base_dir=base_dir,  
        batch_size=32,  
        num_workers=16,  
        seed=seed  
    )  
    
    # Create random subset
    dataset_size = len(test_loader.dataset)  
    subset_size = dataset_size // 10
    indices = random.sample(range(dataset_size), subset_size)  
    subset_dataset = Subset(test_loader.dataset, indices)  
    eval_loader = DataLoader(subset_dataset, batch_size=32, shuffle=False)  
    
    # Initialize model  
    generator = ResnetGenerator2(  
        input_shape=(9, 128, 128),  
        output_channels=1,  
        dim=64  
    ).to(device)  
    
    # Load checkpoint  
    checkpoint_path = 'ckpts/best_model.pth'  
    checkpoint = torch.load(checkpoint_path, map_location=device)  
    generator.load_state_dict(checkpoint['generator_state_dict'])  
    
    # Create save directory  
    save_dir = 'evaluation_results_'  
    os.makedirs(save_dir, exist_ok=True)  
    
    # Print model size  
    model_size = print_model_size(generator)  
    
    # Measure inference speed  
    sample_input = torch.randn(1, 9, 128, 128).to(device)  
    fps = measure_inference_speed(generator, sample_input, device)  
    
    # Evaluate model  
    print("\nCalculating metrics...")  
    metrics = evaluate(generator, eval_loader, device, save_dir)  
    
    # Save results  
    results = {  
        'Model Size (MiB)': f"{model_size:.2f}",  
        'Inference Speed (FPS)': f"{fps:.2f}",  
        **metrics  
    }  
    
    # Print and save results  
    print("\nEvaluation Results:")  
    for metric, value in results.items():  
        print(f"{metric}: {value}")  
    
if __name__ == '__main__':  
    main()