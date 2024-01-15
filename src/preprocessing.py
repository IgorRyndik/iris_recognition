import os
import cv2
import numpy as np
from segment import segment

def create_segmented_images(dataset_dir):
    # Iterate through the subfolders (user IDs)
    for user_id in os.listdir(dataset_dir):
        user_dir = os.path.join(dataset_dir, user_id)
        
        print(user_id)
        image_number = 0
        
        # Iterate through image files in the hand folder
        for image_file in os.listdir(user_dir):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(user_dir, image_file)
                
                # Read the grayscale image
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                ciriris, cirpupil, imwithnoise = segment(gray, 80, True)
                if ciriris is not None:
                    # # Convert circle coordinates to integers
                    # circles = np.round(ciriris).astype("int")

                    x = ciriris[1]
                    y = ciriris[0]
                    r = ciriris[2]
                    
                    iris_crop = gray[y - r:y + r, x - r:x + r]
                    # Display the cropped iris region
                    #cv2.imshow("Cropped Iris", iris_crop)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    # Save the cropped iris image

                    save_dir_path = os.path.join(script_dir, '..', 'cropped', user_id)
                    if not os.path.exists(save_dir_path):
                        os.mkdir(save_dir_path  )

                    save_file_path = os.path.join(save_dir_path, f"{image_number}.png")
                    cv2.imwrite(save_file_path, iris_crop)
                    image_number += 1

if __name__ == '__main__':
     # Get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify the relative path to the dataset directory
    dataset_dir = os.path.join(script_dir, '..', 'data')
    create_segmented_images(dataset_dir)