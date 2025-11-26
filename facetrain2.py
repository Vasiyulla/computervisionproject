import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO
import os

# ------------------ Constants & Setup ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECOGNITION_THRESHOLD = 1.0  # Stricter threshold. Tune this value (0.8-1.0 is a good range).

print(f"[INFO] Running on device: {DEVICE}")

# ------------------ Model Loading ------------------
# MTCNN for precise face alignment and cropping
mtcnn = MTCNN(
    image_size=160, 
    margin=20, 
    device=DEVICE
)

# InceptionResnetV1 for generating face embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# YOLO for fast, initial face detection in the video stream
yolo_face_detector = YOLO("yolov8n-face.pt")

# Dictionary to store known face embeddings {name: embedding}
known_faces = {}

# ------------------ Helper Functions ------------------
def enhance_image(img_bgr):
    """Enhances image for low-light conditions using LAB color space and CLAHE."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB_BGR)

def add_face(name, image_path):
    """
    Loads a single image, generates an embedding, and adds it to the known_faces dictionary.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path not found: {image_path}")
        return

    print(f"[INFO] Processing enrollment image for {name}...")
    try:
        img = Image.open(image_path).convert("RGB")
        face_tensor = mtcnn(img)
        
        if face_tensor is not None:
            face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = resnet(face_tensor).cpu().numpy()
            known_faces[name] = embedding
            print(f"[SUCCESS] Added face for {name}.")
        else:
            print(f"[FAILED] No face detected in the image for {name}.")
            
    except Exception as e:
        print(f"[ERROR] Could not process image {image_path}: {e}")

def recognize_face(face_img):
    """
    Recognizes a face by comparing its embedding with the known faces.
    Returns the best match name and the distance.
    """
    if face_img.size == 0:
        return "Unknown", float('inf')

    try:
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(face_img_rgb)
        
        face_tensor = mtcnn(img_pil)
        if face_tensor is None:
            return "Unknown", float('inf')

        face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()

        min_dist = float('inf')
        best_match = "Unknown"

        for name, known_emb in known_faces.items():
            dist = np.linalg.norm(embedding - known_emb)
            if dist < min_dist:
                min_dist = dist
                if dist < RECOGNITION_THRESHOLD:
                    best_match = name
        
        return best_match, min_dist
    except Exception:
        return "Unknown", float('inf')

# ------------------ Main Execution ------------------
def main():
    # 1. Add the face you want to recognize from a single file
    add_face("Vasiyullah", "25.jpg")  # <-- IMPORTANT: Change this to your image file
    add_face("Danish", "31.jpg")  # <-- IMPORTANT: Change this to your image file
    
    # You can add more people the same way
    # add_face("Another_Person", "another_person.jpg")

    if not known_faces:
        print("[FATAL] No known faces were loaded. Exiting.")
        return

    # 2. Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FATAL] Could not open camera.")
        return

    print("[INFO] Starting video stream. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Detect faces using YOLO
        results = yolo_face_detector.predict(frame, imgsz=320, conf=0.5, verbose=False)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Crop the face from the frame
                face_img = frame[y1:y2, x1:x2]

                # 4. Recognize the cropped face
                name, distance = recognize_face(face_img)
                
                # 5. Draw bounding box and display the name and distance
                # This helps in tuning the threshold!
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = f"{name} ({distance:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Video stream stopped.")

if __name__ == "__main__":
    main()