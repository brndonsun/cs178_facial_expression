import numpy as np

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from collections import defaultdict

def convert_data():
    # Load CSV and prepare file paths
    csv_path = './data/legend.csv'
    image_dir = './images'
    df = pd.read_csv(csv_path)
    df['filepath'] = df['image'].apply(lambda x: os.path.join(image_dir, x))

    # Encode labels
    label_encoder = LabelEncoder()
    df['emotion'] = df['emotion'].str.upper()
    df['encoded_label'] = label_encoder.fit_transform(df['emotion'])
    classes = label_encoder.classes_ 

    # Load images and labels into NumPy arrays
    filepaths = df['image'].values

    images = []
    label_out = []
    filepath_out = []
    for filepath in filepaths:
        # Load and preprocess image
        image_filepath = "./images/" + filepath
        image = Image.open(image_filepath)
        if image.mode == 'RGB':
            image = image.convert('L')  # 'L' mode is for grayscale

        image_array = np.array(image)
        if image_array.shape != (350, 350):
            continue

        image = image.resize((50, 50), Image.LANCZOS)
        image_array = np.array(image).reshape(-1)
        images.append(image_array)  # Convert Tensor to NumPy array
        new_df = df[df['filepath'] == image_filepath]

        # Extract the emotion index
        emotion = new_df['emotion'].values[0]  # Get the emotion value
        emotion_index = label_encoder.transform([emotion])[0]  # Get the encoded index of the emotion
        label_out.append(emotion_index)

        filepath_out.append(image_filepath)
    
    images = np.array(images, dtype=np.float32)  # Convert list to NumPy array
    labels = np.array(label_out, dtype=np.int32)   # Convert labels to NumPy array
    filepaths = np.array(filepath_out, dtype=str)

    # Shuffle the dataset
    indices = np.arange(len(images))  # Create an array of indices
    np.random.shuffle(indices)  # Shuffle the indices in-place
    images = images[indices]  # Apply the shuffled indices to images
    labels = labels[indices]  # Apply the shuffled indices to labels

    return images, labels, classes

if __name__ == '__main__':
    images, labels, classes = convert_data()