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

def compute_flow_metrics(predicted, target):
    """
    Computes MSE, 1-Pixel Error (1PE), and 3-Pixel Error (3PE) between predicted and target tensors.
    
    Args:
    - predicted (torch.Tensor): The predicted flow maps the model.
    - target (torch.Tensor): The ground truth flow maps from RAFT.
    
    Returns:
    - mse (float): Mean Squared Error.
    - pixel_error_1 (float): 1-Pixel Error.
    - pixel_error_3 (float): 3-Pixel Error.
    """
    target = target[0]
    # Ensure the tensors are on the same device
    if predicted.device != target.device:
        target = target.to(predicted.device)

    # Assuming that the first channel represents the actual depth value
    pred_depth = predicted[:, 0, :, :]  # Shape: (N, H, W)
    target_depth = target[:, 0, :, :]    # Shape: (N, H, W)

    # Calculate MSE
    mse = torch.mean((pred_depth - target_depth) ** 2).item()
    
    # Calculate 1-Pixel Error
    # 1PE is the number of pixels where the prediction differs from the target
    one_pixel_error = (pred_depth != target_depth).float()  # Creates a binary tensor
    pixel_error_1 = one_pixel_error.sum().item() / (one_pixel_error.numel())  # Normalized by total number of pixels
    
    # Calculate 3-Pixel Error
    # 3PE considers 3x3 neighborhoods
    target_depth = target_depth.unsqueeze(1)  # Add a channel dimension for convolution
    pred_depth = pred_depth.unsqueeze(1)
    
    # Create a 3x3 kernel
    kernel = torch.ones(1, 1, 3, 3, device=pred_depth.device)
    
    # Use convolution to count the number of matching pixels in a 3x3 neighborhood
    conv_target = torch.nn.functional.conv2d(target_depth, kernel, padding=1)
    conv_predicted = torch.nn.functional.conv2d(pred_depth, kernel, padding=1)
    
    # Calculate 3PE
    pixel_error_3 = (conv_target != conv_predicted).float().sum().item() / (target_depth.numel())
    
    return mse, pixel_error_1, pixel_error_3

def compute_rmse(pred, target):
    """
    Compute the Root Mean Square Error (RMSE) between the predicted and target depth maps.
    
    Args:
        pred (torch.Tensor): Predicted depth map.
        target (torch.Tensor): Target depth map.
    
    Returns:
        float: RMSE value.
    """
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    return rmse.item()  # Convert to a scalar

def compute_accuracy(pred, target, threshold=1.25):
    """
    Compute the accuracy metric: the percentage of predictions that are within the threshold.
    
    Args:
        pred (torch.Tensor): Predicted depth map.
        target (torch.Tensor): Target depth map.
        threshold (float): Threshold value for accuracy calculation.
    
    Returns:
        float: Accuracy value.
    """
    # Compute the ratio of predicted to target depth
    ratio = pred / target

    # Compute accuracy metrics
    acc = (ratio < threshold).float().mean()  # Accuracy for threshold < 1.25
    acc_squared = (ratio < threshold ** 2).float().mean()  # Accuracy for threshold < 1.25²

    return acc.item(), acc_squared.item()  # Convert to scalars

# Example usage
def evaluate_depth_metrics(pred_depth, target_depth):
    # Ensure both tensors are on the same device
    device = pred_depth.device
    max_depth = 100
    min_depth = 0.1
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * pred_depth
    pred_depth = 1 / scaled_disp
    pred_depth = (pred_depth - min_depth) / (max_depth - min_depth)
    target_depth = target_depth.to(device)

    # Compute RMSE
    rmse = compute_rmse(pred_depth, target_depth)

    # Compute accuracy metrics
    acc_125, acc_125_squared = compute_accuracy(pred_depth, target_depth)

    return rmse, acc_125, acc_125_squared

def eval_Depth_Annotations():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = '/home/user/krishnanm0/data/cityscape/val'
    depth_annotations_dir = '/home/user/krishnanm0/data/depth_annotation'
    
    vis_seqpath = []
    anno_seqpath = []
    print("loading paths")
    for city_folder in sorted(os.listdir(val_dir)):
            city_path = os.path.join(val_dir, city_folder)
            if os.path.isdir(city_path):
                frames = sorted(os.listdir(city_path))
                for i in range(len(frames) - 1):
                    if i==20:
                         vis_seqpath.append(os.path.join(city_path, frames[i]))
                         anno_frame = frames[i].split(".")[0] + "_depth." + frames[i].split(".")[1]
                         anno_seqpath.append(os.path.join(depth_annotations_dir,city_folder,anno_frame))
    print("loading images")
    vis_seq = []
    for path in vis_seqpath:
        # Load the images
        frame1 = Image.open(path)
        # Apply any preprocessing or augmentation
        transform = get_augmentations_normal()
        if transform:
            frame1 = transform(frame1)
        vis_seq.append(frame1.to(device).unsqueeze(0))

    anno_seq = []
    for path in anno_seqpath:
        # Load the images
        
        frame1 = Image.open(path)
        # Apply any preprocessing or augmentation
        transform = get_augmentations_normal()
        if transform:
            frame1 = transform(frame1)
        anno_seq.append(frame1.to(device).unsqueeze(0))    
    
    print("loading models")
    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
    for param in resnet.parameters():
                param.requires_grad = False
    # motion_encoder = MotionEncoder(resnet=resnet).to(device)
    depth_model = DepthEstimationModel(resnet=resnet).to(device)
    optimizer_depth = torch.optim.Adam(depth_model.parameters(), lr=1e-4)
    scheduler_depth = torch.optim.lr_scheduler.StepLR(optimizer_depth, step_size=10, gamma=0.5)
    depth_model, optimizer_depth, scheduler_depth, _, _ = load_checkpoint2(depth_model, optimizer_depth, scheduler_depth, "/home/user/krishnanm0/project_checkpoints/depth_model_checkpoint_epoch_8.pth.tar")
    # flow_model = OpticalFlowModel(MotionEncoder=motion_encoder).to(device)
    total_rmse = 0.0
    total_acc_125 = 0.0
    total_acc_125_sq = 0.0 
    count = 0
    print("Calculating losses")
    for (img, anno) in zip(vis_seq, anno_seq):
         depth_model.eval()
         with torch.no_grad():
              depth_map = depth_model(img)
              rmse, acc_125, acc_125_squared = evaluate_depth_metrics(depth_map[0], anno)
              total_rmse+=rmse
              total_acc_125+=acc_125
              total_acc_125_sq+=acc_125_squared
              count+=1
    print("total_rmse:" , total_rmse/count)
    print("total_acc_125:", total_acc_125/count)
    print("total_acc_125_sq:", total_acc_125_sq/count)

def eval_flow_annotations():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dir = '/home/user/krishnanm0/data/cityscape/val'
    
    vis_seqpath = []
    print("loading paths")
    vis_seqpath = []
    count = 0
    for city_folder in sorted(os.listdir(val_dir)):
            city_path = os.path.join(val_dir, city_folder)
            if os.path.isdir(city_path):
                frames = sorted(os.listdir(city_path))
                for i in range(len(frames) - 1):
                    if(i==20 or i==21):
                        vis_seqpath.append(os.path.join(city_path, frames[i]))
                    
    print("loading images")
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

    
    print("loading models")
    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
    for param in resnet.parameters():
                param.requires_grad = False
    raft = models.optical_flow.raft_small(weights='DEFAULT').to(device)
    motion_encoder = MotionEncoder(resnet=resnet).to(device)
    optimizer_motion = torch.optim.Adam(motion_encoder.parameters(), lr=1e-4)
    scheduler_motion = torch.optim.lr_scheduler.StepLR(optimizer_motion, step_size=10, gamma=0.1)

    flow_model = OpticalFlowModel(MotionEncoder=motion_encoder).to(device)
    optimizer_flow = torch.optim.Adam(flow_model.decoder.parameters(), lr=1e-4)
    scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, step_size=10, gamma=0.5)

    total_mse = 0.0
    total_pixel_error_1 = 0.0
    total_pixel_error_3 = 0.0
    count = 0 
    print("Calculating losses")
    for i in range(0,len(vis_seq)-1,2):
        images, target_images = vis_seq[i],vis_seq[i+1]
        # Forward pass 
        pred_flow = flow_model(images, target_images)
        target_flow = raft(images,target_images)
        mse, pixel_error_1, pixel_error_3 = compute_flow_metrics(pred_flow, target_flow)
        total_mse+=mse
        total_pixel_error_1+=pixel_error_1
        total_pixel_error_3+=pixel_error_3
        count+=1
    print("total_mse:" , total_mse/count)
    print("total_pixel_error_1:", total_pixel_error_1/count)
    print("total_pixel_error_3:", total_pixel_error_3/count)

if __name__ == "__main__":
    
    eval_Depth_Annotations()
    eval_flow_annotations() 
    