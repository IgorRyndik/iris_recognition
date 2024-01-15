import os
import cv2


def resize_image(dataset_dir):
    # Iterate through the subfolders (user IDs)
    for user_id in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, user_id)
        
        print(user_id)
        image_number = 0
        
        # Iterate through image files in the hand folder
        for image_file in os.listdir(user_dir):
            if image_file.endswith(".png"):
                image_path = os.path.join(user_dir, image_file)
                
                # Read the grayscale image
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                gray = cv2.resize(gray, (200, 200))

                # Get the current directory of the Python script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                save_dir_path = os.path.join(script_dir, '..', 'cropped', user_id)
                if not os.path.exists(save_dir_path):
                    os.mkdir(save_dir_path  )

                save_file_path = os.path.join(save_dir_path, f"{image_number}.png")
                cv2.imwrite(save_file_path, gray)
                image_number += 1

if __name__ == '__main__':
     # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the dataset directory
    dataset_dir = os.path.join(script_dir, '..', 'cropped')
    resize_image(dataset_dir)
