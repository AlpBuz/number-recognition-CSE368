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


@main_routes.route('/', methods=['GET', 'POST'])
def show_homePage():
    return render_template('index.html')



@main_routes.route('/uploadImage')
def upload_image():
    image = request.files.get('image')

    if image == None:
        return jsonify({"errorMessage": "no image was sent"}), 400



    return jsonify({}), 200