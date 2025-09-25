import cv2
import os

# Haar Cascade path
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Input (uncleaned) and Output (cleaned) dataset folders
input_folder = os.path.join("..", "data")

print("Looking inside:", os.path.abspath(input_folder))
        
output_folder = "clean_data" 

# Create output folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over each student folder
for student_name in os.listdir(input_folder):
    student_input_path = os.path.join(input_folder, student_name)
    student_output_path = os.path.join(output_folder, student_name)

    # Only process if it's a folder (skip files)
    if os.path.isdir(student_input_path):
        if not os.path.exists(student_output_path):
            os.makedirs(student_output_path)

        # Loop through each image in student's folder
        for img_name in os.listdir(student_input_path):
            img_path = os.path.join(student_input_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue  # skip if not an image

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )

            # Crop and resize each detected face
            for idx, (x, y, w, h) in enumerate(faces):
                face_crop = img[y:y+h, x:x+w]

                # Resize face to 224x224 for CNN
                face_resized = cv2.resize(face_crop, (224, 224))

                # Save cropped & resized image in student's folder
                save_name = f"{os.path.splitext(img_name)[0]}_face{idx+1}.jpg"
                save_path = os.path.join(student_output_path, save_name)
                cv2.imwrite(save_path, face_resized)

print("Cleaning complete! Cropped & resized faces saved in 'clean_data/'")
