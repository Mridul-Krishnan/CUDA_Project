import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm  # Import tqdm for progress visualization
from torch.cuda.amp import autocast, GradScaler
import os
import cv2
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


from datasets import CityscapesDataset
from utils.augmentation import get_augmentations, get_augmentations_normal
from models.motion_encoder import MotionEncoder
from models.depth_estimation import DepthEstimationModel
from models.ego_estimation import EgoMotionModel
from models.flow_estimation import OpticalFlowModel
from utils.loss_functions import photometric_loss, regularization_loss, compute_total_loss
from utils.model_utils import save_checkpoint, load_checkpoint, load_checkpoint2
from utils.visuals import *
from utils.loss_functions import generate_warp

def get_training_loaders(train_dir, val_dir, batch_size=16, num_workers=4):
    # Create training dataset and dataloader
    train_dataset = CityscapesDataset(root_dir=train_dir, transform=get_augmentations())
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    # Create validation dataset and dataloader
    val_dataset = CityscapesDataset(root_dir=val_dir, transform=get_augmentations())
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

TBOARD_LOGS = os.path.join(os.getcwd(), "tboard_logs", "training_logs")
if not os.path.exists(TBOARD_LOGS):
    os.makedirs(TBOARD_LOGS)
shutil.rmtree(TBOARD_LOGS)
writer = SummaryWriter(TBOARD_LOGS)

def train_model(resnet, motion_encoder, depth_model, ego_model, flow_model, train_loader, 
                val_loader, num_epochs, device, intrinsic_matrix, checkpoint_dir="checkpoints", 
                vis_seq = [], train_flags=None, freeze_after=None):
    
    if train_flags is None:
        train_flags = {
            'motion_encoder': True,
            'depth_model': True,
            'ego_model': True,
            'flow_model': True
        }
        
    if freeze_after is None:
        freeze_after = {
            'motion_encoder': None,
            'depth_model': None,
            'ego_model': None,
            'flow_model': None
        }
        
    resnet.train()
    motion_encoder.train()
    depth_model.train()
    ego_model.train()
    flow_model.train()
    
    # Define optimizers
    if not train_flags['motion_encoder']:
        load_checkpoint(motion_encoder, None, None, checkpoint_path=checkpoint_dir, model_name="motion_encoder")
        motion_encoder.eval()
    if not train_flags['depth_model']:
        load_checkpoint(depth_model, None, None, checkpoint_path=checkpoint_dir, model_name="depth_model")
        depth_model.eval()
    if not train_flags['ego_model']:
        load_checkpoint(ego_model, None, None, checkpoint_path=checkpoint_dir, model_name="ego_model")
        ego_model.eval()
    if not train_flags['flow_model']:
        load_checkpoint(flow_model, None, None, checkpoint_path=checkpoint_dir, model_name="flow_model")
        flow_model.eval()

    # Define optimizers and schedulers only for models that are being trained
    if train_flags['motion_encoder']:
        optimizer_motion = torch.optim.Adam(motion_encoder.parameters(), lr=1e-4)
        scheduler_motion = torch.optim.lr_scheduler.StepLR(optimizer_motion, step_size=10, gamma=0.1)
    if train_flags['depth_model']:
        optimizer_depth = torch.optim.Adam(depth_model.parameters(), lr=1e-4)
        scheduler_depth = torch.optim.lr_scheduler.StepLR(optimizer_depth, step_size=10, gamma=0.5)
    if train_flags['ego_model']:
        optimizer_ego = torch.optim.Adam(ego_model.decoder.parameters(), lr=1e-4)
        scheduler_ego = torch.optim.lr_scheduler.StepLR(optimizer_ego, step_size=10, gamma=0.5)
    if train_flags['flow_model']:
        optimizer_flow = torch.optim.Adam(flow_model.decoder.parameters(), lr=1e-4)
        scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=10, gamma=0.5)
        
    scaler = GradScaler()

    start_epoch = 11
    global_iteration = 0
    
    continueTrain = True
    createContinueGif = True
    if continueTrain:
        
        motion_encoder, optimizer_motion, scheduler_motion, _, _ = load_checkpoint2(motion_encoder, optimizer_motion, scheduler_motion, "/home/user/krishnanm0/project_checkpoints3/motion_encoder_checkpoint_epoch_10.pth.tar")
        ego_model = EgoMotionModel(MotionEncoder=motion_encoder).to(device)
        flow_model = OpticalFlowModel(MotionEncoder=motion_encoder).to(device)
        optimizer_ego = torch.optim.Adam(ego_model.decoder.parameters(), lr=1e-4)
        scheduler_ego = torch.optim.lr_scheduler.StepLR(optimizer_ego, step_size=10, gamma=0.5)
        optimizer_flow = torch.optim.Adam(flow_model.decoder.parameters(), lr=1e-4)
        scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=10, gamma=0.5)
        depth_model, optimizer_depth, scheduler_depth, _, _ = load_checkpoint2(depth_model, optimizer_depth, scheduler_depth, "/home/user/krishnanm0/project_checkpoints3/depth_model_checkpoint_epoch_10.pth.tar")
        ego_model, optimizer_ego, scheduler_ego, _, _ = load_checkpoint2(ego_model, optimizer_ego, scheduler_ego, "/home/user/krishnanm0/project_checkpoints3/ego_model_checkpoint_epoch_10.pth.tar")
        flow_model, optimizer_flow, scheduler_flow, _, _ = load_checkpoint2(flow_model, optimizer_flow, scheduler_flow, "/home/user/krishnanm0/project_checkpoints3/flow_model_checkpoint_epoch_10.pth.tar")
        for i, param_group in enumerate(optimizer_motion.param_groups):
            param_group['lr'] = 1e-4
        for i, param_group in enumerate(optimizer_depth.param_groups):
            param_group['lr'] = 1e-4
        for i, param_group in enumerate(optimizer_ego.param_groups):
            param_group['lr'] = 1e-4
        for i, param_group in enumerate(optimizer_flow.param_groups):
            param_group['lr'] = 1e-4
        if createContinueGif:
            depth_sequence = []
            ego_motion_sequence = []
            flow_sequence = []
            warp_depth = []
            warp_flow = []
            normal_sequence = []
            with torch.no_grad():
                for i in range(len(vis_seq)-1):
                    images, target_images = vis_seq[i],vis_seq[i+1]
                    # Forward pass
                    pred_depth = depth_model(images)
                    ego_motion = ego_model(images, target_images)  
                    pred_flow = flow_model(images, target_images)
                    warp = generate_warp(images,pred_depth[0],pred_flow, ego_motion,intrinsic_matrix)
                    warp_depth.append(warp[0])
                    warp_flow.append(warp[1])
                    depth_sequence.append(pred_depth[0])
                    flow_sequence.append(pred_flow)
                    ego_motion_sequence.append(ego_motion)
                    normal_sequence.append(images)

            # Create GIFs for each visualization
            create_depth_gif(depth_sequence, output_file=os.path.join(checkpoint_dir,f"depth_map{start_epoch-1}.gif"))
            create_ego_motion_gif(ego_motion_sequence, output_file=os.path.join(checkpoint_dir,f"ego_motion{start_epoch-1}.gif"))
            create_optical_flow_gif(flow_sequence, output_file=os.path.join(checkpoint_dir,f"optical_flow{start_epoch-1}.gif"))
            create_rgb_gif(warp_depth, output_file=os.path.join(checkpoint_dir,f"warp_depth_map{start_epoch-1}.gif"))
            create_rgb_gif(warp_flow, output_file=os.path.join(checkpoint_dir,f"warp_flow_map{start_epoch-1}.gif"))
            create_rgb_gif(normal_sequence, output_file=os.path.join(checkpoint_dir,f"normal{start_epoch-1}.gif"))


    for epoch in range(start_epoch,num_epochs):
        # Training phase
        if train_flags['depth_model']:
            depth_model.train()
        if train_flags['ego_model']:
            ego_model.train()
        if train_flags['flow_model']:
            flow_model.train()
            
            
        if freeze_after['motion_encoder'] and epoch >= freeze_after['motion_encoder']:
            print("Freezing motion encoder after epoch", epoch)
            for param in motion_encoder.parameters():
                param.requires_grad = False
        if freeze_after['depth_model'] and epoch >= freeze_after['depth_model']:
            print("Freezing depth model after epoch", epoch)
            for param in depth_model.parameters():
                param.requires_grad = False
        if freeze_after['ego_model'] and epoch >= freeze_after['ego_model']:
            print("Freezing ego motion model after epoch", epoch)
            for param in ego_model.parameters():
                param.requires_grad = False
        if freeze_after['flow_model'] and epoch >= freeze_after['flow_model']:
            print("Freezing flow model after epoch", epoch)
            for param in flow_model.parameters():
                param.requires_grad = False
                   
        running_loss = 0.0
        # Progress bar for training
        train_loader_iter = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Training)", leave=False)

        for _, batch in enumerate(train_loader_iter):
            images = batch[0].to(device)  # Source images
            target_images = batch[1].to(device)  # Target images (next frames)

            if train_flags['motion_encoder']:
                optimizer_motion.zero_grad()
            if train_flags['depth_model']:
                optimizer_depth.zero_grad()
            if train_flags['ego_model']:
                optimizer_ego.zero_grad()
            if train_flags['flow_model']:
                optimizer_flow.zero_grad()
            
            with autocast():
                # Forward pass for depth, ego-motion, and flow models
                
                pred_depth = depth_model(images) if train_flags['depth_model'] else None
                ego_motion = ego_model(images, target_images) if train_flags['ego_model'] else None
                pred_flow = flow_model(images, target_images) if train_flags['flow_model'] else None

                total_loss, loss_reg_depth, loss_photo_depth, loss_reg_flow, loss_photo_flow = compute_total_loss(pred_depth, pred_flow, images, target_images, ego_motion, intrinsic_matrix)

            # Backward pass and optimization
            scaler.scale(total_loss).backward()
            running_loss += total_loss.item()

            # Update tqdm progress bar with current loss
            train_loader_iter.set_postfix(loss=total_loss.item())
            writer.add_scalar(f'Iter/Total_Loss', total_loss.item(), global_step=global_iteration)
            writer.add_scalar(f'Iter/loss_reg_depth', loss_reg_depth.item(), global_step=global_iteration)
            writer.add_scalar(f'Iter/loss_photo_depth', loss_photo_depth.item(), global_step=global_iteration)
            writer.add_scalar(f'Iter/loss_reg_flow', loss_reg_flow.item(), global_step=global_iteration)
            writer.add_scalar(f'Iter/loss_photo_flow', loss_photo_flow.item(), global_step=global_iteration)
            writer.add_scalars(f'Iter/Comb_Loss', {
                    'loss_reg_depth': loss_reg_depth.item(),
                    'loss_photo_depth': loss_photo_depth.item(),
                    'loss_reg_flow': loss_reg_flow.item(),
                    'loss_photo_flow': loss_photo_flow.item()
                }, global_step = global_iteration)
            global_iteration+=1
            
            # break


        ## Add Tensor board logs
        writer.add_scalar(f'Total_Loss/Train', total_loss.item(), global_step=epoch)
        writer.add_scalar(f'Epoch/loss_reg_depth', loss_reg_depth.item(), global_step=epoch)
        writer.add_scalar(f'Epoch/loss_photo_depth', loss_photo_depth.item(), global_step=epoch)
        writer.add_scalar(f'Epoch/loss_reg_flow', loss_reg_flow.item(), global_step=epoch)
        writer.add_scalar(f'Epoch/loss_photo_flow', loss_photo_flow.item(), global_step=epoch)
        
        # Validation phase
        motion_encoder.eval()
        depth_model.eval()
        ego_model.eval()
        flow_model.eval()
        
        val_loss = 0.0
        validate = False
        # Progress bar for validation
        
        if validate:
            val_loader_iter = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] (Validation)", leave=False)
            with torch.no_grad():
                for batch in val_loader_iter:
                    images = batch[0].to(device)  # Source images
                    target_images = batch[1].to(device)  # Target images (next frames)

                    # Forward pass
                    pred_depth = depth_model(images)
                    ego_motion = ego_model(images, target_images)  
                    pred_flow = flow_model(images, target_images)

                    # Compute validation loss
                    total_loss, _, _, _, _ = compute_total_loss(pred_depth, pred_flow, images, target_images, ego_motion, intrinsic_matrix)
                    val_loss += total_loss.item()
                    # break

            val_loss /= len(val_loader)
            writer.add_scalar(f'Total_Loss/Valid', val_loss, global_step=epoch)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        if train_flags['motion_encoder']:
            scaler.step(optimizer_motion)
        if train_flags['depth_model']:
            scaler.step(optimizer_depth)
        if train_flags['ego_model']:
            scaler.step(optimizer_ego)
        if train_flags['flow_model']:
            scaler.step(optimizer_flow)
        # Learning rate scheduler step
        scheduler_motion.step()
        scheduler_depth.step()
        scheduler_ego.step()
        scheduler_flow.step()
        scaler.update()
        
                
        depth_sequence = []
        ego_motion_sequence = []
        flow_sequence = []
        warp_depth = []
        warp_flow = []
        normal_sequence = []
        with torch.no_grad():
            for i in range(len(vis_seq)-1):
                images, target_images = vis_seq[i],vis_seq[i+1]
                # Forward pass
                pred_depth = depth_model(images)
                ego_motion = ego_model(images, target_images)  
                pred_flow = flow_model(images, target_images)
                warp = generate_warp(images,pred_depth[0],pred_flow, ego_motion,intrinsic_matrix)
                warp_depth.append(warp[0])
                warp_flow.append(warp[1])
                depth_sequence.append(pred_depth[0])
                flow_sequence.append(pred_flow)
                ego_motion_sequence.append(ego_motion)
                normal_sequence.append(images)

        # Create GIFs for each visualization
        create_depth_gif(depth_sequence, output_file=os.path.join(checkpoint_dir,f"depth_map{epoch}.gif"))
        create_ego_motion_gif(ego_motion_sequence, output_file=os.path.join(checkpoint_dir,f"ego_motion{epoch}.gif"))
        create_optical_flow_gif(flow_sequence, output_file=os.path.join(checkpoint_dir,f"optical_flow{epoch}.gif"))
        create_rgb_gif(warp_depth, output_file=os.path.join(checkpoint_dir,f"warp_depth_map{epoch}.gif"))
        create_rgb_gif(warp_flow, output_file=os.path.join(checkpoint_dir,f"warp_flow_map{epoch}.gif"))
        create_rgb_gif(normal_sequence, output_file=os.path.join(checkpoint_dir,f"normal{epoch}.gif"))
        
        print(checkpoint_dir)
        # Save model checkpoints
        if train_flags['motion_encoder']:
            save_checkpoint(motion_encoder, optimizer_motion, scheduler_motion, epoch, running_loss, "motion_encoder", checkpoint_dir=checkpoint_dir)
        if train_flags['depth_model']:
            save_checkpoint(depth_model, optimizer_depth, scheduler_depth, epoch, running_loss, "depth_model", checkpoint_dir=checkpoint_dir)
        if train_flags['ego_model']:
            save_checkpoint(ego_model, optimizer_ego, scheduler_ego, epoch, running_loss, "ego_model", checkpoint_dir=checkpoint_dir)
        if train_flags['flow_model']:
            save_checkpoint(flow_model, optimizer_flow, scheduler_flow, epoch, running_loss, "flow_model", checkpoint_dir=checkpoint_dir)
            
        if (epoch + 1) % 10 == 0:  # Save additional checkpoint every 10 epochs
            torch.save({
                'epoch': epoch + 1,
                'motion_enc_state_dict': motion_encoder.state_dict(),
                'depth_model_state_dict': depth_model.state_dict(),
                'ego_model_state_dict': ego_model.state_dict(),
                'flow_model_state_dict': flow_model.state_dict(),
                'motion_enc_state_dict': optimizer_motion.state_dict(),
                'optimizer_depth_state_dict': optimizer_depth.state_dict(),
                'optimizer_ego_state_dict': optimizer_ego.state_dict(),
                'optimizer_flow_state_dict': optimizer_flow.state_dict(),
                'scheduler_motion_state_dict': scheduler_motion.state_dict(),
                'scheduler_depth_state_dict': scheduler_depth.state_dict(),
                'scheduler_ego_state_dict': scheduler_ego.state_dict(),
                'scheduler_flow_state_dict': scheduler_flow.state_dict(),
                'loss': running_loss / len(train_loader)
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth.tar'))
        

    print('Training completed.')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dir = '/home/user/krishnanm0/data/cityscape/train'
    val_dir = '/home/user/krishnanm0/data/cityscape/val'

    train_loader, val_loader = get_training_loaders(train_dir, val_dir)
    
    vis_seqpath = []
    for city_folder in sorted(os.listdir(val_dir)):
            city_path = os.path.join(val_dir, city_folder)
            if os.path.isdir(city_path):
                frames = sorted(os.listdir(city_path))
                for i in range(len(frames) - 1):
                    vis_seqpath.append(os.path.join(city_path, frames[i]))
                break
    vis_seq = []
    for path in vis_seqpath:
        # Load the images
        frame1 = cv2.imread(path)
        # Convert from BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        # Convert to PIL images for PyTorch transforms
        frame1 = Image.fromarray(frame1)
        # Apply any preprocessing or augmentation
        transform = get_augmentations_normal()
        if transform:
            frame1 = transform(frame1)
        vis_seq.append(frame1.to(device).unsqueeze(0))
    # print(torch.min(vis_seq[0]),torch.max(vis_seq[0]))
    image_width = 512
    image_height = 256
    fx = 2262.52 * image_width / 2048
    fy = 2262.52 * image_height / 1024

    intrinsic_matrix = torch.tensor([
    [fx, 0, image_width / 2],
    [0, fy, image_height / 2],
    [0, 0, 1]]).float().to(device)

    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
    motion_encoder = MotionEncoder(resnet=resnet).to(device)
    depth_model = DepthEstimationModel(resnet=resnet).to(device)
    ego_model = EgoMotionModel(MotionEncoder=motion_encoder).to(device)
    flow_model = OpticalFlowModel(MotionEncoder=motion_encoder).to(device)

    train_model(resnet, motion_encoder, depth_model, ego_model, flow_model, train_loader, val_loader, num_epochs=50, device=device, intrinsic_matrix=intrinsic_matrix, checkpoint_dir='/home/user/krishnanm0/project_checkpoints3', vis_seq=vis_seq)