from facenet_pytorch import MTCNN
import torch
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import euclidean
import json
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm

mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()  # Load the pre-trained model

def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.squeeze()  # or embedding1.squeeze()
    embedding2 = embedding2.squeeze() 
    embedding1 = embedding1 / norm(embedding1)
    embedding2 = embedding2 / norm(embedding2)
    return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
def create_jfile():
    face_database = {}
    training_image_dir = r"C:\Users\epals\OneDrive\Desktop\C_vision_p1\face_database"

    for img_file in os.listdir(training_image_dir):
        image_path = os.path.join(training_image_dir, img_file)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, probs = mtcnn.detect(rgb_image)

        if boxes is not None and probs[0] > 0.9:  # Ensure good confidence
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                face = rgb_image[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (160, 160))
                face_tensor = torch.Tensor(face_resized).permute(2, 0, 1).unsqueeze(0)
                face_tensor = (face_tensor - 127.5) / 128.0

                face_embedding = model(face_tensor).detach().numpy().tolist()
                person_name = img_file.split('.')[0]

                face_database[person_name] = face_embedding

    with open('face_database.json', 'w') as f:
        json.dump(face_database, f)
def load_face_database():
    json_file_path = 'face_database.json'
    
    # Ask user if they want to use the loaded database or create a new one
    x = int(input("Use loaded database (1) or create new one (2)? "))

    if x == 1:
        # Check if the JSON file exists
        if not os.path.exists(json_file_path):
            print(f"{json_file_path} does not exist. Creating new database...")
            create_jfile()  # Call function to create the JSON database
        else:
            # Try to load the JSON file and check for validity
            try:
                with open(json_file_path, 'r') as f:
                    face_database = json.load(f)
                
                # Ensure the JSON is not empty or malformed
                if not face_database:
                    print(f"{json_file_path} is empty or malformed. Creating new database...")
                    create_jfile()
                else:
                    print(f"Successfully loaded {json_file_path}.")
                    return face_database
            
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading {json_file_path}: {e}. Creating new database...")
                create_jfile()

    elif x == 2:
        print("Creating new face database...")
        create_jfile()  # Call function to create the JSON database

    # Load the database after creation
    with open(json_file_path, 'r') as f:
        face_database = json.load(f)

    return face_database

image_dir = r"C:\Users\epals\OneDrive\Desktop\C_vision_p1\face_compare_test"  # Path to your single image file

image_files = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, file) for file in image_files]

im_c = 0
face_database = load_face_database()

for image_file in image_files:
    # Load the image once at the beginning of the loop
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    if image is None:
        print(f"Error loading image: {image_file}, either no face was detected or no face was present.")
        continue  # Skip to the next image if it fails to load
    mf = False  # Flag to indicate if a match was found

    matched_names = []  # List to store matched person names
    # Detect faces in the image
    boxes, _ = mtcnn.detect(rgb_image)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]  # Get bounding box coordinates
            
            # Ensure coordinates are within image bounds
            h, w, _ = rgb_image.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 - x1 > 0 and y2 - y1 > 0:  # Ensure valid size for the face region
                face = rgb_image[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (160, 160))  # Resize the face for model input
                
                # Convert to tensor, normalize, and generate the face embedding
                face_tensor = torch.Tensor(face_resized).permute(2, 0, 1).unsqueeze(0)
                face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
                face_embedding = model(face_tensor).detach().numpy()
                
                # Compare the face with the stored embeddings
                for person_name, stored_embedding in face_database.items():
                    stored_embedding = np.array(stored_embedding)
                    
                    # Calculate the cosine similarity
                    similarity = cosine_similarity(face_embedding, stored_embedding)

                    # Set a threshold for face matching
                    threshold = 0.65
                    if similarity >= threshold:
                        print(f"Match found: {person_name} with similarity {similarity}")
                        matched_names.append(person_name)
                        mf = True  # Mark that a match has been found

                        # Draw a bounding box and label the face in the original image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 20)  # Green bounding box
                        cv2.putText(image, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 11, (255, 255, 255), 30)  # White label text
            else:
                print(f"Invalid face region for {image_file}. Skipping.")

    # Save the image if a match was found
    if mf:
        im_c += 1  # Increment the counter for unique filenames
        ou_na = matched_names[-1] if matched_names else "unknown"
        output_path = os.path.join("output", ou_na +  "_" + str(im_c) + ".jpg")
        cv2.imwrite(output_path, image)  # Save the image with the bounding box and label
        print(f"Image saved as {output_path}")
        
        
        
        
        
#note: for the json file to get name correct name the database file the persons name ie Jhon.jpeg will have a name Jhon.