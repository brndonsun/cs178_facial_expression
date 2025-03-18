import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from collections import defaultdict

def convert_data():
    # Load CSV and prepare file paths
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Load CSV and prepare file paths
    csv_path = os.path.join(current_dir, 'data', 'legend.csv')
    image_dir = os.path.join(current_dir, 'images')
    df = pd.read_csv(csv_path)
    
    # Print sample of the original data
    print("First few rows of the CSV:")
    print(df.head())
    
    # Check for the 'image' column
    if 'image' not in df.columns:
        print("Error: 'image' column not found in CSV.")
        print("Available columns:", df.columns.tolist())
        return None, None, None, None
    
    # Create filepath column
    df['filepath'] = df['image'].apply(lambda x: os.path.join(image_dir, x))
    
    # Print sample paths for debugging
    print("\nSample filepaths in DataFrame:")
    for i in range(min(3, len(df))):
        print(f"  Image: {df['image'].iloc[i]}")
        print(f"  Filepath: {df['filepath'].iloc[i]}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['emotion'] = df['emotion'].str.upper()
    df['encoded_label'] = label_encoder.fit_transform(df['emotion'])
    classes = label_encoder.classes_
    print(f"\nEncoded emotion classes: {classes}")

    # Load images and labels into NumPy arrays
    filepaths = df['image'].values

    images = []
    label_out = []
    filepath_out = []
    
    # Debug counter
    success_count = 0
    error_count = 0
    
    for i, filepath in enumerate(filepaths):
        if i < 3 or i % 100 == 0:  # Print details for first few and every 100th
            print(f"\nProcessing image {i+1}/{len(filepaths)}: {filepath}")
        
        # Construct the image filepath
        image_filepath = os.path.join(image_dir, filepath)
        
        try:
            # Check if file exists before trying to open it
            if not os.path.exists(image_filepath):
                print(f"  File does not exist: {image_filepath}")
                error_count += 1
                continue
                
            image = Image.open(image_filepath)
            
            # Print image details for debugging
            if i < 3:
                print(f"  Original image mode: {image.mode}, size: {image.size}")
            
            if image.mode == 'RGB':
                image = image.convert('L')  # 'L' mode is for grayscale

            image_array = np.array(image)
            if image_array.shape != (350, 350):
                if i < 3 or i % 100 == 0:
                    print(f"  Skipping image with incorrect shape: {image_array.shape}")
                error_count += 1
                continue

            image = image.resize((50, 50), Image.LANCZOS)
            image_array = np.array(image).reshape(50, 50, 1)
            
            # Find the corresponding emotion by matching the filename
            # Look for exact matches to the filename
            matching_rows = df[df['image'] == filepath]
            
            if len(matching_rows) == 0:
                print(f"  Error: No matching row found for image: {filepath}")
                print("  This suggests a mismatch between CSV entries and image filenames")
                error_count += 1
                continue
                
            emotion = matching_rows['emotion'].values[0]
            emotion_index = label_encoder.transform([emotion])[0]
            
            images.append(image_array)
            label_out.append(emotion_index)
            filepath_out.append(image_filepath)
            
            success_count += 1
            
        except Exception as e:
            print(f"  Error processing {image_filepath}: {e}")
            error_count += 1
    
    print(f"\nProcessing summary:")
    print(f"  Successfully processed: {success_count} images")
    print(f"  Errors/skipped: {error_count} images")
    
    if success_count == 0:
        print("No images were successfully processed!")
        return None, None, None, None
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(label_out, dtype=np.int32)
    filepaths = np.array(filepath_out, dtype=str)

    # Shuffle the dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    filepaths = filepaths[indices]

    return images, labels, classes, filepaths


if __name__ == '__main__':
    print("Starting data conversion...")
    images, labels, classes, filepaths = convert_data()
    
    if images is not None:
        seed = 42
        Xtrain, Xval, Ytrain, Yval = train_test_split(images, labels, test_size=0.2, random_state=seed)
        print(f"Training set shape: {Xtrain.shape}")
        print(f"Validation set shape: {Xval.shape}")
        print(f"Classes: {classes}")
    else:
        print("Data conversion failed. Check the errors above.")
