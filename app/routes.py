from flask import *
import os

# Blueprint allows you to organize routes
main_routes = Blueprint('main', __name__)



@main_routes.route('/', methods=['GET', 'POST'])
def show_homePage():
    return render_template('index.html')



@main_routes.route('/uploadImage')
def upload_image():


    return