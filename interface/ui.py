import tempfile
from flask import Flask, render_template, request
import os
import cv2
import base64
from database import Database
from models.altered_xception import AlteredXception
from models.rmac_model import RMACModel
from models.faster_r_cnn import FasterRCNN
from utils import DATABASE_PATH, DOG_DB_PATH, GAMES_DB_PATH, ART_DB_PATH, LECTURE_DB_PATH
from models.query import query_image
import sys
# Allows model, database, and utils modules to be accessed 
# from inside the interface directory
sys.path.append('./')

app = Flask(__name__)
app.config['SECRET_KEY'] = "sdfpAWkfs[aldpas"

@app.before_first_request
def initializeDBandModels() -> None:
    '''Before we start the website, load the models and the database to be ready for query images.'''
    global DATABASE, AX_MODEL, RMAC_MODEL, RCNN_MODEL, DOG_DB, GAMES_DB, ART_DB, LECTURE_DB
    DATABASE = Database(DATABASE_PATH)
    DOG_DB = Database(DOG_DB_PATH)
    GAMES_DB = Database(GAMES_DB_PATH)
    ART_DB = Database(ART_DB_PATH)
    LECTURE_DB = Database(LECTURE_DB_PATH)
    AX_MODEL = AlteredXception(database=DATABASE)
    RMAC_MODEL = RMACModel(AX_MODEL.model.get_layer("conv2d_3").output_shape[1:], 3, DATABASE)
    RCNN_MODEL = FasterRCNN(DATABASE)

@app.route('/', methods=['GET', 'POST'])
def initialize_ui():
    b64_preview_image = None
    top_yt_links = None
    if request.method == 'POST':
        # Get image input from user and check that it's not None (i.e. no image sent)
        imagefile = request.files['imagefile'].read()
        im_request_file = request.files.get('imagefile', '')
        if not im_request_file:
            # Invalid, reload page
            return render_template('index.html', b64_preview_image=b64_preview_image, yt_links=top_yt_links)
        
        # Determine models selected to compute the images
        curr_model_select = request.form.get("compute_models")
        curr_models = []
        if curr_model_select != "ALL":
            curr_models.append(globals()[curr_model_select])
        else:
            curr_models = [AX_MODEL, RMAC_MODEL, RCNN_MODEL]

        # Determine database to query from, set the model's DB to this DB
        curr_database = globals()[request.form.get("database")]
        for model in curr_models:
            model.database = curr_database

        # Create temporary image to compute predictions, save to base 64 for preview, and delete after user leaves session or submits new query
        ext = '.' + im_request_file.filename.split('.')[1]
        with tempfile.NamedTemporaryFile(suffix=ext, mode='w+b', dir="interface/static/images", delete=True) as temp_img_file:
            temp_img_file.write(imagefile)
            image_file_url = os.path.relpath(temp_img_file.name)
            # Read, save, and encode image in base64 to be returned to the user
            curr_image = cv2.imread(image_file_url)
            buffer = cv2.imencode('.jpg', curr_image)[1]
            b64_image = base64.b64encode(buffer).decode('utf-8')
            b64_preview_image = f'data:image/jpg;base64, {b64_image}'
            # Query the models and get each model's guess at the most related YouTube Videos
            top_yt_links = query_image(temp_img_file.name, curr_models, 10)
    return render_template('index.html', b64_preview_image=b64_preview_image, yt_links=top_yt_links)


if __name__ == "__main__":
    app.run(debug=True)
