from flask import *
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from app.AI.src.network import Net



# Blueprint allows you to organize routes
main_routes = Blueprint('main', __name__)
AIModel = Net()
AIModel.load_state_dict(torch.load('app/AI/results/model.pth'))  
AIModel.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@main_routes.route('/', methods=['GET'])
def show_homePage():
    return render_template('index.html')



@main_routes.route('/uploadImage', methods=['POST'])
def upload_image():
    try:
        file = request.files.get('image')
        if file == None:
            print("no image was sent", flush=True)
            return jsonify({"errorMessage": "no image was sent"}), 400
        
        uploaded_path = os.path.join("app/static/uploads", file.filename)
        print(uploaded_path, flush=True)
        file.save(uploaded_path)
        image = Image.open(file)
        input = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = AIModel(input)
            prediction = torch.argmax(output, dim=1).item()  

        print(prediction, flush=True)
        return jsonify({'prediction': prediction, "filePath": f'static/uploads/{file.filename}'}), 200

    except Exception as e:
        print(e, flush=True)
        return jsonify({"errorMessage": "a server error has occured"}), 500