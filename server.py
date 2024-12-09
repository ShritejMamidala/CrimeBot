#IMPORTS
from fastapi import FastAPI, WebSocket
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles
from torchvision import transforms
import torch
import os
import json
from openai import OpenAI


#custom models we trained for face attribution/feature extraction
from models import MultiTaskModel # race/age/gender
from models2 import FaceAttributeModel #28 possible features

# preprocessing for first model - based on how it was trained
preprocess_model1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# fpr second model
preprocess_model2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# FastAPI server
app = FastAPI()

#frontend directory folder for the webpage that includes the html/css/js
frontend_dir = "frontend"
if not os.path.exists(frontend_dir):
    print(f"Error: Frontend directory '{frontend_dir}' does not exist.")
else:
    app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")
    print(f"Frontend directory '{frontend_dir}' mounted successfully.")

# use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load first model
model1 = MultiTaskModel()
state_dict_path_model1 = r"C:\Users\shrit\Desktop\Ml_Projects\Ml_Projects\Notebooks\best_face_age_gender_race_extraction_state_dict2.pth"
state_dict_model1 = torch.load(state_dict_path_model1, map_location=device)
model1.load_state_dict(state_dict_model1)
model1.to(device)
model1.eval()
print("Face feature extraction model (Model 1) loaded successfully.")

# load second model
model2 = FaceAttributeModel(num_features=30)
state_dict_path_model2 = r"C:\\Users\\shrit\\Desktop\\Ml_Projects\\Ml_Projects\\pytorch_results_weights\\best_model_state_dict.pth"
state_dict_model2 = torch.load(state_dict_path_model2, map_location=device)
model2.load_state_dict(state_dict_model2)
model2.to(device)
model2.eval()
print("Second feature extraction model (Model 2) loaded successfully.")

# YOLO model to detect faces in frames sent
yolo_model = YOLO(r"C:\\Users\\shrit\\Desktop\\Ml_Projects\\Ml_Projects\\Notebooks\\runs\\detect\\train9\\weights\\best.pt")
print("YOLO model loaded successfully.")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#for gpt summary at end

existing_boxes = {}  # currently active bounding boxes
unmatched_frames = {}  # how long each ID has been unmatched
finalized_feature_vectors = {}  # holds finalized feature vectors
iou_threshold = 0.3  # IOU threshold for matching detections
max_unmatched_frames = 5  # how long a frame can be unmatched before removing the id
sent_to_gpt_ids = set()  # holds what IDs already sent to GPT


# IOU calculation
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # threshold calculations
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0


def scale_bbox(bbox, scale_factor, img_width, img_height):
    """Scale the bounding box ___ while keeping it within webcam image"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    new_width = width * scale_factor
    new_height = height * scale_factor

    center_x = x1 + width / 2
    center_y = y1 + height / 2

    new_x1 = max(0, center_x - new_width / 2)
    new_y1 = max(0, center_y - new_height / 2)
    new_x2 = min(img_width, center_x + new_width / 2)
    new_y2 = min(img_height, center_y + new_height / 2)

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def assign_ids(detections, existing_boxes, threshold=0.5):
    """Assign IDs to detections based on IOU"""
    global unmatched_frames  
    matched_ids = set()  # holds which IDs are matched in this frame
    assigned_ids = []

    for box in detections:
        matched = False
        for existing_id, existing_box in existing_boxes.items():
            iou = calculate_iou(box, existing_box)
            if iou > threshold:
                assigned_ids.append((box, existing_id))
                existing_boxes[existing_id] = box  # update box
                matched_ids.add(existing_id)
                matched = True
                break
        if not matched:
            new_id = len(existing_boxes) + 1
            assigned_ids.append((box, new_id))
            existing_boxes[new_id] = box
            matched_ids.add(new_id)

    #  unmatched boxes logic
    for existing_id in list(existing_boxes.keys()):
        if existing_id not in matched_ids:
            unmatched_frames[existing_id] = unmatched_frames.get(existing_id, 0) + 1
            if unmatched_frames[existing_id] > max_unmatched_frames:
                del existing_boxes[existing_id]
                del unmatched_frames[existing_id]
        else:
            unmatched_frames.pop(existing_id, None)  # Reset if matched

    return assigned_ids

@app.websocket("/process") #accept websocket to get frames
async def process_frames(websocket: WebSocket):
    await websocket.accept()
    
    #  labels for model outputs to intrepret predictions/proabbailites of model output
    age_labels = {
        '10-19': 0,
        '20-29': 1,
        '30-39': 2,
        '40-49': 3,
        '50-59': 4,
    }
    gender_labels = {"Male": 1, "Female": 0}
    race_labels = {
        "East Asian": 0,
        "Indian": 1,
        "Black": 2,
        "White": 3,
        "Middle Eastern": 4,
        "Latino_Hispanic": 5
    }
    model2_feature_labels = [
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Chubby', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Mustache', 'Narrow_Eyes',
        'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
        'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie'
    ]

    while True:
        # getting the frame data and making it compatible
        data = await websocket.receive_bytes()
        image = Image.open(BytesIO(data)).convert("RGB")
        image_array = np.array(image)
        img_height, img_width, _ = image_array.shape

        # predict using yolo to find faces
        results = yolo_model.predict(source=image_array, save=False, conf=0.5)
        detections = []
        for result in results:
            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  #  bbox coordinates
                bbox = scale_bbox(bbox, 1.5, img_width, img_height)  # scale bbox by 1.5
                confidence = box.conf[0].item()  # conf score
                detections.append({"bbox": bbox, "confidence": confidence})

        # assign IDs Based on if IOU intersects little enough from frame to frame
        assigned_detections = assign_ids(
            [det["bbox"] for det in detections], existing_boxes, iou_threshold
        )
        processed_results = []

        # logic to deal with every detection yolo makes 
        for detection, assigned_id in assigned_detections:
            x1, y1, x2, y2 = map(int, detection)
            confidence = next(
                det["confidence"] for det in detections if det["bbox"] == detection
            )

            # skipping recalculating feature vectors for IDs already created - dont want to run logic on the same person multiple times
            if assigned_id in finalized_feature_vectors:
                processed_results.append({
                    "bbox": detection,
                    "confidence": confidence,
                    "id": assigned_id,
                })

                # GPT for new IDs
                if assigned_id not in sent_to_gpt_ids:
                    gpt_prompt = prepare_gpt_prompt({assigned_id: finalized_feature_vectors[assigned_id]})
                    print(f"Sending to GPT for ID {assigned_id}:")
                    print(gpt_prompt)
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": gpt_prompt}],
                        model="gpt-3.5-turbo",
                    )
                    gpt_response = response.choices[0].message.content
                    print(f"GPT Response for ID {assigned_id}: {gpt_response}")

                    # Send GPT response to the frontend
                    await websocket.send_json({
                        "type": "gpt_response",
                        "id": assigned_id,
                        "content": gpt_response,
                    })

                    sent_to_gpt_ids.add(assigned_id)  # record gpt prompt for that id as sent
                continue


            # sent bbox/confidence score/id to frontend (all detections), ensures user/client sees the program working
            processed_results.append({
                "bbox": detection,
                "confidence": confidence,
                "id": assigned_id,
            })

            # preprocess and evaluate if confidence > 80% - run pipeline only on good images
            if confidence > 0.8:
                cropped_face = image.crop((x1, y1, x2, y2))  # crop it to the bbox to use for predictions

                # run preprocess function and evaluate with Model 1
                preprocessed_face_model1 = preprocess_model1(cropped_face).unsqueeze(0)
                with torch.no_grad():
                    age_out, gender_out, race_out = model1(preprocessed_face_model1.to(device))
                feature_vectors_model1 = {
                    "age": {label: round(torch.softmax(age_out, dim=1).squeeze().tolist()[idx], 3) for label, idx in age_labels.items()},
                    "gender": {label: round(torch.sigmoid(gender_out).item() if idx == 0 else 1 - torch.sigmoid(gender_out).item(), 3) for label, idx in gender_labels.items()},
                    "race": {label: round(torch.softmax(race_out, dim=1).squeeze().tolist()[idx], 3) for label, idx in race_labels.items()},
                } # this adds the age/gender/race feature vectors (predictions) to a dict

                # run second preprocess function and evaluate with Model 2
                preprocessed_face_model2 = preprocess_model2(cropped_face).unsqueeze(0)
                with torch.no_grad():
                    second_model_out = model2(preprocessed_face_model2.to(device))
                second_model_out_list = second_model_out.squeeze().tolist()
                top_features = sorted(
                    [(model2_feature_labels[i], round(value, 3)) for i, value in enumerate(second_model_out_list)],
                    key=lambda x: x[1], reverse=True
                )
                feature_vectors_model2 = {
                    label: value for label, value in top_features[:10] if value > 0.3
                } # this adds the age/gender/race feature vectors (predictions) to a dict

                #  store full feature vectors together into 1
                finalized_feature_vectors[assigned_id] = {
                    "Model 1": feature_vectors_model1,
                    "Model 2": feature_vectors_model2,
                }

        # bbox Data to front
        await websocket.send_json({"type": "detection_results", "data": processed_results})

        #  log feature vectors
        print(f"Finalized Feature Vectors: {finalized_feature_vectors}")


 #function to prompt gpt giving it the feature vectors and then whenever the func is called the prompt can be sent to  front end to be displayed
def prepare_gpt_prompt(feature_vectors):
    prompt = """
            Analyze the following facial features and provide a brief, concise summary of the person's physical characteristics as if describing them for a police report.
            Refer to the person as "the suspect" throughout.
            Include only one specific age range (e.g., 10-19, not 10-39) and specify one race accurately.
            Focus solely on describing physical features.
"""    
    
    for id, vectors in feature_vectors.items():
        prompt += f"ID {id}:\n"
        prompt += f"Model 1 (Age, Gender, Race): {json.dumps(vectors['Model 1'], indent=2)}\n"
        prompt += f"Model 2 (Attributes): {json.dumps(vectors['Model 2'], indent=2)}\n\n"
    return prompt

#DONE!!!