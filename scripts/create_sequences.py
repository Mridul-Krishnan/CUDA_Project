"""
    This script processes the Cityscapes dataset to create sequences of images from city folders.
    Each sequence consists of exactly 30 images. The script organizes these sequences into 'test', 'train',
    and 'val' subfolders based on the specified dataset splits.

    Key functionalities:
    1. Create a list of city folders for each dataset split (e.g., 'test', 'train', 'val').
    2. Identify sequences of 30 images within each city folder.
    3. Copy valid sequences into an output directory, organized by split.
    4. Skip processing of a sequence if its output folder already exists to avoid redundant work.

    Usage:
    - Ensure that the dataset path is correctly set to the Cityscapes dataset location.
    - Adjust the `base_output_path` to the desired output directory for the extracted sequences.
    - Run the script to process the dataset and generate the sequences.
"""
import os
import shutil
# from dotenv import load_dotenv

# # Load environment variables for the dataset base_path
# load_dotenv()

# Base output path for the extracted sequences
base_output_path = '/home/user/krishnanm0/data/cityscape'

def create_city_folder_list(dataset_splits, dataset_path='//home/nfs/inf6/data/datasets/cityscapes_new/leftImg8bit_sequence'):
    city_folders = []
    for split in dataset_splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            city_folders.extend([os.path.join(split_path, city) for city in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, city))])

    return city_folders

def create_sequences():
    dataset_splits = ['train', 'test', 'val']
    
    # Process each dataset split
    for split in dataset_splits:
        # Create list of all the cities in the current split
        city_folders = create_city_folder_list([split])
        
        # Process each city folder to find sequences with 30 images
        for city_folder in city_folders:
            sequences = {}

            for file_name in os.listdir(city_folder):
                if file_name.endswith('.png'):
                    # Extract the sequence identifier (cityname_sequenceNum)
                    sequence_id = '_'.join(file_name.split('_')[:2])
                    if sequence_id not in sequences:
                        sequences[sequence_id] = []
                    sequences[sequence_id].append(file_name)

            # Filter to find sequences with exactly 30 images
            valid_sequences = {seq: files for seq, files in sequences.items() if len(files) == 30}
            # print(valid_sequences)
            # Copy each valid sequence to a new subfolder in the output directory under the respective split
            for sequence, image_files in valid_sequences.items():
                # Define the output path for the current split
                split_output_path = os.path.join(base_output_path, split, sequence)
                
                # Check if the sequence folder already exists
                if os.path.exists(split_output_path):
                    print(f"Sequence folder '{split_output_path}' already exists. Skipping to the next sequence.")
                    continue  # Skip the current sequence and move to the next one
                
                print(f"Creating sequence folder: {split_output_path}")
                
                # Create the directory for the sequence
                os.makedirs(split_output_path, exist_ok=True)
                
                # Copy the sequence images to the output subfolder
                for image_file in image_files:
                    src_path = os.path.join(city_folder, image_file)
                    dst_path = os.path.join(split_output_path, image_file)
                    shutil.copy(src_path, dst_path)

            print(f"Sequences of '{city_folder.split('/')[-1]}' for split '{split}' complete.")
    print("All sequences with 30 images have been processed.")

create_sequences()
