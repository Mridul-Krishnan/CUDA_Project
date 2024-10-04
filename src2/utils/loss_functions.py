import torch
import torch.nn.functional as F
from utils.ego_warp_utils import *
import matplotlib.pyplot as plt

# image_width = 512
# image_height = 256
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # Generate a meshgrid of pixel coordinates (in homogeneous form)
# y, x = torch.meshgrid(torch.arange(0, image_width, device=device), 
#                         torch.arange(0, image_height, device=device),  indexing='ij')
# ones = torch.ones_like(x)
# pixel_coords = torch.stack([x, y, ones], dim=0).float()  # shape: [3, height, width]
# pixel_coords = pixel_coords.view(3, -1)  # shape: [3, num_pixels]

# # Intrinsic inverse
# image_width = 512
# image_height = 256
# fx = 2262.52 * image_width / 2048
# fy = 1096.98 * image_height / 1024

# intrinsic_matrix = torch.tensor([
# [fx, 0, image_width / 2],
# [0, fy, image_height / 2],
# [0, 0, 1]]).float().to(device)
# intrinsic_inv = torch.inverse(intrinsic_matrix)

# Photometric Loss: SSIM + L1
def photometric_loss(pred, target, alpha=0.85):
    def ssim(x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim_map = SSIM_n / SSIM_d
        return torch.clamp((1 - ssim_map) / 2, 0, 1)

    ssim_loss = ssim(pred, target)
    l1_loss = torch.abs(pred - target)

    return alpha * ssim_loss.mean() + (1 - alpha) * l1_loss.mean()

# Regularization Loss for Depth and Flow
def regularization_loss(pred, image):
    def gradient_x(img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    pred_x = gradient_x(pred)
    pred_y = gradient_y(pred)
    
    img_x = gradient_x(image)
    img_y = gradient_y(image)

    reg_loss_x = torch.abs(pred_x) * torch.exp(-torch.abs(img_x))
    reg_loss_y = torch.abs(pred_y) * torch.exp(-torch.abs(img_y))

    return reg_loss_x.mean() + reg_loss_y.mean()

def regularization_loss_flow(pred_flow, image):
    def gradient_x(img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    pred_x = gradient_x(pred_flow)
    pred_y = gradient_y(pred_flow)

    # Calculate gradients for the image (3 channels)
    img_x = gradient_x(image)
    img_y = gradient_y(image)

    # Regularization loss calculation
    reg_loss_x = torch.abs(pred_x) * torch.exp(-torch.abs(img_x[:, :2, :, :]))  # Use only the first two channels for consistency
    reg_loss_y = torch.abs(pred_y) * torch.exp(-torch.abs(img_y[:, :2, :, :]))  # Same here

    return reg_loss_x.mean() + reg_loss_y.mean()

def resize_depth_maps(depth_maps, target_size = (256, 512)):
    """
    Resizes all depth maps in the list to the target size.
    Args:
    - depth_maps (list of tensors): List of depth maps at different scales.
    - target_size (tuple): Target size to resize all depth maps to (H, W).

    Returns:
    - List of resized depth maps.
    """
    resized_depth_maps = [F.interpolate(depth, size=target_size, mode='bilinear', align_corners=True) 
                          for depth in depth_maps]
    return resized_depth_maps    

def warp_image_with_depth(source_img, depth, ego_motion, intrinsic):
    batch_size, _, height, width = source_img.size()
    
    
    depth = torch.clamp(depth, min = 1e-3)   # Replace zero depth values with a small positive constant


    # Generate a meshgrid of pixel coordinates (in homogeneous form)
    y, x = torch.meshgrid(torch.arange(0, height, device=source_img.device), 
                          torch.arange(0, width, device=source_img.device),  indexing='ij')
    # print("mesh", y.shape)
    ones = torch.ones_like(x)
    pixel_coords = torch.stack([x, y, ones], dim=0).float()  # shape: [3, height, width]
    pixel_coords = pixel_coords.view(3, -1)  # shape: [3, num_pixels]

    # Intrinsic inverse
    intrinsic_inv = torch.inverse(intrinsic)

    # Compute 3D coordinates in camera frame
    depth = depth.view(batch_size, 1, -1)  # shape: [B, 1, num_pixels]
    
    # Ensure cam_coords has a batch dimension
    cam_coords = intrinsic_inv @ pixel_coords  # shape: [3, num_pixels]
    cam_coords = cam_coords.unsqueeze(0).expand(batch_size, -1, -1)  # shape: [B, 3, num_pixels]

    # Multiply cam_coords by depth, should be broadcastable
    cam_coords = cam_coords * depth  # shape: [B, 3, num_pixels]
    
    # Apply ego-motion (rotation and translation)
    ego_motion = pose_vec2mat(ego_motion, rotation_mode='euler') # [R|t] matrix
    transformed_coords = ego_motion[:, :3, :3] @ cam_coords + ego_motion[:, :3, 3].unsqueeze(2)

    # Project back to 2D
    pixel_coords_warped = intrinsic @ transformed_coords  # shape: [B, 3, num_pixels]
    pixel_coords_warped = pixel_coords_warped[:, :2, :] / torch.clamp(pixel_coords_warped[:, 2, :].unsqueeze(1), min=1e-8)
    # Normalize pixel coordinates for grid sampling
    pixel_coords_warped = pixel_coords_warped.view(batch_size, 2, height, width)
    pixel_coords_warped = pixel_coords_warped.permute(0, 2, 3, 1)  # [B, H, W, 2]
    
    # print("pre-normal coords", torch.min(pixel_coords_warped[:, :, :, 1]),torch.max(pixel_coords_warped[:, :, :, 1]))
    # Normalize to range [-1, 1] for F.grid_sample
    pixel_coords_warped[:, :, :, 0] = (pixel_coords_warped[:, :, :, 0] / width - 0.5) * 2
    pixel_coords_warped[:, :, :, 1] = (pixel_coords_warped[:, :, :, 1] / height - 0.5) * 2
    # Warp the source image using grid sampling
    # print("src", torch.min(source_img),torch.max(source_img))
    # print("coords", torch.min(pixel_coords_warped),torch.max(pixel_coords_warped))
    
    warped_image = F.grid_sample(source_img, pixel_coords_warped, mode='bilinear', padding_mode='border', align_corners=True)


    # mask_x = (pixel_coords_warped[:, :, :, 0] >= -1) & (pixel_coords_warped[:, :, :, 0] <= 1)
    # mask_y = (pixel_coords_warped[:, :, :, 1] >= -1) & (pixel_coords_warped[:, :, :, 1] <= 1)
    # mask = mask_x & mask_y  # Combine x and y masks

    # # Warp the source image using grid sampling
    # warped_image = F.grid_sample(source_img, pixel_coords_warped, mode='bilinear', padding_mode='border', align_corners=True)

    # # Apply mask to the warped image
    # mask = mask.unsqueeze(1).float()  # Reshape mask for broadcasting
    # warped_image = warped_image * mask 

    return warped_image


def flow_warp(img, flow):
    """
    Warps the source image using the predicted optical flow.
    """
    B, C, H, W = img.shape
    _, _, H_flow, W_flow = flow.shape

    i_range = torch.arange(0, W).view(1, 1, -1).expand(B, H, W).to(img.device)
    j_range = torch.arange(0, H).view(1, -1, 1).expand(B, H, W).to(img.device)

    flow_u = flow[:, 0, :, :]
    flow_v = flow[:, 1, :, :]

    i_range = i_range.float() + flow_u
    j_range = j_range.float() + flow_v

    i_range = 2.0 * i_range / (W - 1) - 1.0
    j_range = 2.0 * j_range / (H - 1) - 1.0

    # grid = torch.stack([i_range, j_range], dim=-1).permute(0, 2, 3, 1)
    grid = torch.stack([i_range, j_range], dim=-1)
    warped_img = F.grid_sample(img, grid, align_corners=True)
    return warped_img

def disp_to_depth(disp, min_depth = 0.1, max_depth = 100):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    depth = []
    for i in range(len(disp)):
        scaled_disp = min_disp + (max_disp - min_disp) * disp[i]
        depth.append(1 / scaled_disp)
    return  depth

def compute_total_loss(pred_depths, pred_flow, images, target_images, ego_motion, intrinsic_matrix, alpha=0.85):
    '''pred_depth[0], pred_flow, images, target_images, ego_motion, intrinsic_matrix
    return total_loss, lambda_reg_depth*loss_reg_depth, loss_photo_depth, lambda_reg_flow*loss_reg_flow, loss_photo_flow
    '''
    

    # Get the size of the highest resolution depth map (disp0)
    target_size = pred_depths[0].shape[2:]  # (height, width)
    # pred_depths = disp_to_depth(pred_depths)
    # Resize all depth maps to the size of disp0
    pred_depths_resized =pred_depths[:1] + resize_depth_maps(pred_depths[1:], target_size)

    # Define loss weights for each depth scale (higher weight for finer scales)
    scale_weights = [1.0, 1/4, 1/8, 1/16]  

    lambda_reg_flow = 1e-2
    lambda_reg_depth = 1e-2
    # Initialize total loss
    total_depth_loss = 0.0
    
    # Compute regularization loss for each resized depth scale
    for i, depth in enumerate(pred_depths_resized):
        loss_reg_depth = regularization_loss(depth, images)
        total_depth_loss = total_depth_loss + scale_weights[i] * loss_reg_depth 
    loss_reg_depth = total_depth_loss
    warped_img_depth = warp_image_with_depth(images, pred_depths_resized[0], ego_motion, intrinsic_matrix)
    loss_photo_depth = photometric_loss(warped_img_depth, target_images, alpha)

    # 3. Photometric loss for optical flow (warp image using flow)
    warped_img_flow = 0
    loss_photo_flow = 0
    warped_img_flow = flow_warp(images, pred_flow)
    loss_photo_flow = photometric_loss(warped_img_flow, target_images, alpha)

    # 4. Regularization loss for flow (smoothness)
    loss_reg_flow = 0
    loss_reg_flow = regularization_loss_flow(pred_flow, images)
    # Total loss (sum of all components)
    total_loss = lambda_reg_depth*loss_reg_depth + loss_photo_flow + lambda_reg_flow*loss_reg_flow + loss_photo_depth
    return total_loss, lambda_reg_depth*loss_reg_depth, loss_photo_depth, lambda_reg_flow*loss_reg_flow, loss_photo_flow

def generate_warp(images, pred_depth, pred_flow, ego_motion, intrinsic_matrix):
    '''images, pred_depth, pred_flow, ego_motion, intrinsic_matrix'''
    # max_depth = 100
    # min_depth = 0.1
    # min_disp = 1 / max_depth
    # max_disp = 1 / min_depth
    # scaled_disp = min_disp + (max_disp - min_disp) * pred_depth
    # pred_depth = 1 / scaled_disp
    
    warped_img_depth = warp_image_with_depth(images, pred_depth, ego_motion, intrinsic_matrix)
    warped_img_flow = flow_warp(images, pred_flow)

    return (warped_img_depth, warped_img_flow)
