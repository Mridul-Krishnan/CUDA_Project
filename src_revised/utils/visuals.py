import imageio
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import flow_to_image

def visualize_depth(depth_map):
    depth_map = depth_map.squeeze().cpu().detach().numpy()
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label="Depth")
    plt.title("Depth Map")
    plt.show()

def visualize_ego_motion(ego_motion):
    # ego_motion is expected to be a 6D vector
    translation = ego_motion[:, :3]
    rotation = ego_motion[:, 3:]
    
    print("Translation Vector (x, y, z):", translation.cpu().detach().numpy())
    print("Rotation Vector (Euler Angles):", rotation.cpu().detach().numpy())

def visualize_optical_flow(flow):
    flow = flow.squeeze().cpu().detach().numpy()  # Shape: (2, H, W)
    u = flow[0]  # Horizontal flow
    v = flow[1]  # Vertical flow

    # Visualize with quiver plot
    plt.figure(figsize=(10, 10))
    plt.quiver(u, v, angles="xy", scale_units="xy", scale=1, color="r")
    plt.title("Optical Flow (Quiver Plot)")
    plt.show()


def create_depth_gif(depth_sequence, output_file="depth_animation.gif"):
    # depth_sequence: A list of depth maps (each as a torch.Tensor or numpy array)
    
    images = []
    max_depth = 100
    min_depth = 0.1
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    for i, depth_map in enumerate(depth_sequence):
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        
        scaled_disp = min_disp + (max_disp - min_disp) * depth_map
        # depth_map = 1 / scaled_disp
        # depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        
        depth_map = scaled_disp
        depth_map = depth_map.squeeze().cpu().detach().numpy()
        vmax = np.percentile(depth_map, 95)

        ax.imshow(depth_map, cmap='magma',vmax=vmax)
        ax.set_title(f"Depth Map Frame {i}")
        plt.colorbar(ax.imshow(depth_map, cmap='magma',vmax=vmax))
        plt.axis('off')  # Hide the axes for a cleaner look
        # Save the frame
        frame_path = f"depth_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))  # Append to image list for gif creation
        plt.close(fig)  # Close the plot
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)  # fps: frames per second
    print(f"Depth GIF saved at {output_file}")

def create_ego_motion_gif(ego_motion_sequence, output_file="ego_motion_animation.gif"):
    images = []
    
    for i, ego_motion in enumerate(ego_motion_sequence):
        fig, ax = plt.subplots()
        
        translation = ego_motion[:, :3].cpu().detach().numpy()  # Extract translation
        rotation = ego_motion[:, 3:].cpu().detach().numpy()  # Extract rotation
        
        # Create a bar plot for translation and rotation
        ax.bar(["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"], list(translation[0]) + list(rotation[0]))
        ax.set_ylim([-2, 2])  # Set limit to keep scaling consistent
        
        ax.set_title(f"Ego-Motion Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        
        # Save the frame
        frame_path = f"ego_motion_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))
        plt.close('all')
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)
    print(f"Ego-Motion GIF saved at {output_file}")

def create_optical_flow_gif(flow_sequence, output_file="flow_animation.gif"):
    images = []
    
    for i, flow in enumerate(flow_sequence):
        fig, ax = plt.subplots()
        
        flow = flow_to_image(flow)
        flow = flow[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(flow)
        ax.set_title(f"flow Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        
        # Save the frame
        frame_path = f"flow_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))  # Append to image list for gif creation
        plt.close('all')  # Close the plot
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)
    print(f"Optical Flow GIF saved at {output_file}")

def create_rgb_gif(rgb_sequence, output_file="rgb_animation.gif"):
    # rgb_sequence: A list of RGB images (each as a torch.Tensor or numpy array)
    # Standard deviation and mean used for normalization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)

    
    images = []
    for i, rgb_image in enumerate(rgb_sequence):
         # Reverse the normalization
        rgb_image = rgb_image * std + mean
        # Convert to numpy and ensure the correct format
        rgb_image = rgb_image[0].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
        
        
        # Plot and save each frame as an image
        fig, ax = plt.subplots()
        ax.imshow(rgb_image)
        ax.set_title(f"RGB Frame {i}")
        plt.axis('off')  # Hide the axes for a cleaner look
        
        # Save the frame
        frame_path = f"rgb_frame_{i}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))  # Append to image list for gif creation
        plt.close('all')  # Close the plot
        
        # Remove the temporary image
        os.remove(frame_path)
    
    # Save the frames as a GIF
    imageio.mimsave(output_file, images, fps=5)  # fps: frames per second
    print(f"RGB GIF saved at {output_file}")



