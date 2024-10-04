from IPython.display import display, Image as IPImage
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
from PIL import Image
from utils.loss_functions import *
from torchvision.utils import flow_to_image


def visualize_gif(gif_path):
    """
    Display a GIF inline in a Jupyter notebook or similar environment.

    Args:
        gif_path (str): Path to the GIF file to display.
    """
    with open(gif_path, "rb") as f:
        display(IPImage(data=f.read(), format='png'))

def create_optical_flow_gif(flow_sequence, predicted_sequence ,normal_sequence, output_file="flow_animation.gif"):
    images = []
    device = torch.device(flow_sequence[0].device)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    
    for i, (flow, pred, rgb_image) in enumerate(zip(flow_sequence, predicted_sequence ,normal_sequence)):
        fig, ax = plt.subplots()
        
        flow = flow_to_image(flow)
        flow = flow[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(flow)
        # ax.set_title(f"flow Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        
        # Save the frame
        flow_frame_path = f"flow_frame_{i}.png"
        plt.savefig(flow_frame_path)
        
        plt.close(fig)  # Close the plot


        # PREDICTION SEQUENCE
        pred = pred * std + mean
        # Convert to numpy and ensure the correct format
        pred = pred[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)
        # ax.set_title(f"Normal Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        # Save the frame
        pred_frame_path = f"pred_frame_{i}.png"
        plt.savefig(pred_frame_path)
        plt.close('all')  # Close the plot

        # NORMAL SEQUENCE
        rgb_image = rgb_image * std + mean
        # Convert to numpy and ensure the correct format
        rgb_image = rgb_image[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)
        # ax.set_title(f"Normal Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        # Save the frame
        normal_frame_path = f"normal_frame_{i}.png"
        plt.savefig(normal_frame_path)
        plt.close('all')  # Close the plot

        image1 = mpimg.imread(flow_frame_path)
        image2 = mpimg.imread(pred_frame_path)
        image3 = mpimg.imread(normal_frame_path)

        # Create a figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        # Display each image
        axes[0].imshow(image1)
        axes[0].axis('off')  # Hide the axes

        axes[1].imshow(image2)
        axes[1].axis('off')

        axes[2].imshow(image3)
        axes[2].axis('off')
        plt.show()
        frame_path = f"frame_{i}.png"
        plt.savefig(frame_path)
        plt.close('all') 
        images.append(imageio.imread(frame_path))  # Append to image list for gif creation
        # Remove the temporary image
        os.remove(flow_frame_path)
        os.remove(pred_frame_path)
        os.remove(normal_frame_path)
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)
    print(f"Optical Flow GIF saved at {output_file}")
    visualize_gif(output_file)

def create_depth_gif(depth_sequence, predicted_sequence ,normal_sequence, output_file="depth_animation.gif"):
    images = []
    device = torch.device(depth_sequence[0].device)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    max_depth = 100
    min_depth = 0.1
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth

    for i, (depth_map, pred, rgb_image) in enumerate(zip(depth_sequence, predicted_sequence ,normal_sequence)):
        fig, ax = plt.subplots()
        
        scaled_disp = min_disp + (max_disp - min_disp) * depth_map
        depth_map = 1 / scaled_disp
        depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        depth_map = depth_map.squeeze().cpu().detach().numpy()
        
        ax.imshow(depth_map, cmap='gray')
        ax.set_title(f"Depth Map Frame {i}")
        plt.colorbar(ax.imshow(depth_map, cmap='gray'))
        plt.axis('off')  # Hide the axes for a cleaner look
        # Save the frame
        depth_frame_path = f"depth_frame_{i}.png"
        plt.savefig(depth_frame_path)
        
        plt.close(fig)  # Close the plot


        # PREDICTION SEQUENCE
        pred = pred * std + mean
        # Convert to numpy and ensure the correct format
        pred = pred[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)
        # ax.set_title(f"Normal Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        # Save the frame
        pred_frame_path = f"pred_frame_{i}.png"
        plt.savefig(pred_frame_path)
        plt.close('all')  # Close the plot

        # NORMAL SEQUENCE
        rgb_image = rgb_image * std + mean
        # Convert to numpy and ensure the correct format
        rgb_image = rgb_image[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)
        # ax.set_title(f"Normal Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        # Save the frame
        normal_frame_path = f"normal_frame_{i}.png"
        plt.savefig(normal_frame_path)
        plt.close('all')  # Close the plot

        image1 = mpimg.imread(depth_frame_path)
        image2 = mpimg.imread(pred_frame_path)
        image3 = mpimg.imread(normal_frame_path)

        # Create a figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

        # Display each image
        axes[0].imshow(image1)
        axes[0].axis('off')  # Hide the axes

        axes[1].imshow(image2)
        axes[1].axis('off')

        axes[2].imshow(image3)
        axes[2].axis('off')
        plt.show()
        frame_path = f"frame_{i}.png"
        plt.savefig(frame_path)
        plt.close('all') 
        images.append(imageio.imread(frame_path))  # Append to image list for gif creation
        # Remove the temporary image
        os.remove(depth_frame_path)
        os.remove(pred_frame_path)
        os.remove(normal_frame_path)
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)
    print(f"Depth-Ego Estimation GIF saved at {output_file}")
    visualize_gif(output_file)

def load_sequence_images(sequence_folder):
    """
    Load all images from a sequence folder.

    Args:
        sequence_folder (str): Path to the sequence folder containing images.

    Returns:
        list of PIL.Image: List of images loaded from the sequence folder.
    """
    # Get a sorted list of all .png files in the folder
    image_files = sorted([os.path.join(sequence_folder, file) 
                          for file in os.listdir(sequence_folder) if file.endswith('.png')])
    
    # Load each image file as a PIL Image object
    images = [Image.open(img_file) for img_file in image_files]
    return images

def create_gif(depth_model,flow_model,ego_model, vis_seq):

    device = depth_model.device

    image_width = 512
    image_height = 256
    fx = 2262.52 * image_width / 2048
    fy = 2262.52 * image_height / 1024

    intrinsic_matrix = torch.tensor([
    [fx, 0, image_width / 2],
    [0, fy, image_height / 2],
    [0, 0, 1]]).float().to(device)

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
                normal_sequence.append(vis_seq[i+1])
    print("Optical Flow Gif [Flow Map - Full Flow - Original Frame]")
    create_optical_flow_gif(flow_sequence, warp_flow, normal_sequence)
    print("Depth and Ego Estimation Gif [Depth Map - Rigid Flow - Original Frame]")
    create_optical_flow_gif(depth_sequence, warp_depth, normal_sequence)
    
    