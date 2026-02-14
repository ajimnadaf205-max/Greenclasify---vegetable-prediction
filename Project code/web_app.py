from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import numpy as np
from model import get_model
from torchvision import transforms

# Nutritional info (calories and vitamins per 100g)
nutrition_info = {
    "Bean": {"Calories": "31 kcal", "Vitamins": "Vitamin C, K"},
    "Bitter Gourd": {"Calories": "17 kcal", "Vitamins": "Vitamin A, C"},
    "Bottle Gourd": {"Calories": "14 kcal", "Vitamins": "Vitamin C"},
    "Brinjal": {"Calories": "25 kcal", "Vitamins": "Vitamin C, K"},
    "Broccoli": {"Calories": "34 kcal", "Vitamins": "Vitamin C, K"},
    "Cabbage": {"Calories": "25 kcal", "Vitamins": "Vitamin C, K"},
    "Capsicum": {"Calories": "26 kcal", "Vitamins": "Vitamin A, C"},
    "Carrot": {"Calories": "41 kcal", "Vitamins": "Vitamin A, K"},
    "Cauliflower": {"Calories": "25 kcal", "Vitamins": "Vitamin C, K"},
    "Cucumber": {"Calories": "16 kcal", "Vitamins": "Vitamin K"},
    "Papaya": {"Calories": "43 kcal", "Vitamins": "Vitamin A, C"},
    "Potato": {"Calories": "77 kcal", "Vitamins": "Vitamin C, B6"},
    "Pumpkin": {"Calories": "26 kcal", "Vitamins": "Vitamin A, C"},
    "Radish": {"Calories": "16 kcal", "Vitamins": "Vitamin C"},
    "Tomato": {"Calories": "18 kcal", "Vitamins": "Vitamin C, K"}
}


def load_model(num_classes, model_path="vegetable_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


app = Flask(__name__, static_folder="static", template_folder="templates")

# Class names (folder names) and display names
class_names = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
               "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
               "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]
display_names = [name.replace("_", " ") for name in class_names]

model, device = load_model(len(class_names))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert('RGB')
    except Exception:
        return jsonify({"error": "invalid image"}), 400

    # keep image in RGB format for preprocessing
    img_array = np.array(image)
    img_tensor = preprocess_image(img_array).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        topk = torch.topk(probs, k=5)
        top_probs = topk.values.cpu().numpy().tolist()
        top_indices = topk.indices.cpu().numpy().tolist()
        top_classes = [display_names[i] for i in top_indices]

    predictions = [{"class": c, "prob": p} for c, p in zip(top_classes, top_probs)]
    top1 = top_classes[0]
    nutrition = nutrition_info.get(top1, {})

    return jsonify({"predictions": predictions, "nutrition": nutrition})


if __name__ == '__main__':
    # run on 5000 to avoid conflict with Streamlit default port
    app.run(host='0.0.0.0', port=5000, debug=True)
