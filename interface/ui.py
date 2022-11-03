import tempfile
from flask import Flask, render_template, request
# from flask_dropzone import Dropzone
import sys
import os
sys.path.append('..')
from database import Database
from altered_xception import AlteredXception
from rmac_model import RMACModel
from faster_r_cnn import FasterRCNN
from utils import DATABASE_PATH
from query import query_image


app = Flask(__name__)
# dropzone = Dropzone(app)
app.config['SECRET_KEY'] = "sdfpAWkfs[aldpas"

# Dropzone settings
# app.config['DROPZONE_UPLOAD_MULTIPLE'] = False
# app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
# app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
# app.config['DROPZONE_REDIRECT_VIEW'] = 'initialize_ui'

@app.before_first_request
def initializeDBandModels() -> None:
    global DATABASE, AX_MODEL, RMAC_MODEL, RCNN_MODEL
    DATABASE = Database(DATABASE_PATH)
    AX_MODEL = AlteredXception(DATABASE)
    RMAC_MODEL = RMACModel(AX_MODEL.model.get_layer("conv2d_3").output_shape[1:], 3, DATABASE)
    RCNN_MODEL = FasterRCNN(DATABASE)

@app.route('/', methods=['GET', 'POST'])
def initialize_ui():
    file_url = None
    top_yt_links = None
    if request.method == 'POST':
        imagefile = request.files['imagefile'].read()
        im_request_file = request.files.get('imagefile', '')
        ext = '.' + im_request_file.filename.split('.')[1]
        with tempfile.NamedTemporaryFile(suffix=ext, mode='w+b', dir="./static/images", delete=False) as temp_img_file:
            temp_img_file.write(imagefile)
            file_url = os.path.relpath(temp_img_file.name)
            top_yt_links = query_image(temp_img_file.name, [AX_MODEL, RMAC_MODEL, RCNN_MODEL], 10)
    return render_template('index.html', file_url=file_url, yt_links=top_yt_links)

if __name__ == "__main__":
    app.run(debug=True)
