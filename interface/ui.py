import tempfile
from flask import Flask, render_template, request
import sys
import os
import cv2
import base64
sys.path.append('..')
from database import Database
from altered_xception import AlteredXception
from rmac_model import RMACModel
from faster_r_cnn import FasterRCNN
from utils import DATABASE_PATH
from query import query_image


app = Flask(__name__)
app.config['SECRET_KEY'] = "sdfpAWkfs[aldpas"

@app.before_first_request
def initializeDBandModels() -> None:
    global DATABASE, AX_MODEL, RMAC_MODEL, RCNN_MODEL
    DATABASE = Database(DATABASE_PATH)
    AX_MODEL = AlteredXception(database=DATABASE)
    RMAC_MODEL = RMACModel(AX_MODEL.model.get_layer("conv2d_3").output_shape[1:], 3, DATABASE)
    RCNN_MODEL = FasterRCNN(DATABASE)

@app.route('/', methods=['GET', 'POST'])
def initialize_ui():
    b64_preview_image = None
    top_yt_links = None
    if request.method == 'POST':
        imagefile = request.files['imagefile'].read()
        im_request_file = request.files.get('imagefile', '')
        ext = '.' + im_request_file.filename.split('.')[1]
        with tempfile.NamedTemporaryFile(suffix=ext, mode='w+b', dir="./static/images", delete=True) as temp_img_file:
            temp_img_file.write(imagefile)
            image_file_url = os.path.relpath(temp_img_file.name)
            curr_image = cv2.imread(image_file_url)
            buffer = cv2.imencode('.jpg', curr_image)[1]
            b64_image = base64.b64encode(buffer).decode('utf-8')
            b64_preview_image = f'data:image/jpg;base64, {b64_image}'
            top_yt_links = query_image(temp_img_file.name, [AX_MODEL, RMAC_MODEL, RCNN_MODEL], 10)
    return render_template('index.html', b64_preview_image=b64_preview_image, yt_links=top_yt_links)

if __name__ == "__main__":
    app.run(debug=True)
