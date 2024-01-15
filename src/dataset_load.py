import os
import librosa
import numpy as np
import cv2

from save_dataset_to_json import save_dataset_to_json


def load_dataset(dataset_dir):
    # Initialize empty lists to store data and labels
    data = []
    labels = []

    # Iterate through the subfolders (speaker IDs)
    for user_id in os.listdir(dataset_dir):
        speaker_dir = os.path.join(dataset_dir, user_id)
        
        print(user_id)

        # Iterate through the files in each subfolder
        for image_file in os.listdir(speaker_dir):
            
            image_path = os.path.join(speaker_dir, image_file)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(img)
            labels.append(user_id)

  
    np.save("images.npy", data)
    np.save("labels.npy", labels)

if __name__ == "__main__":
   
    # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the dataset directory
    dataset_dir = os.path.join(script_dir, '..', 'cropped')
    load_dataset(dataset_dir)