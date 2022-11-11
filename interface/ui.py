import tempfile
from flask import Flask, render_template, request
import sys
sys.path.append('./')
import os
import cv2
import base64
from database import Database
from models.altered_xception import AlteredXception
from models.rmac_model import RMACModel
from models.faster_r_cnn import FasterRCNN
from utils import DATABASE_PATH, EVAL_DB_PATH, DOG_DB_PATH
from models.query import query_image
from eval.evaluate import load_json_data


app = Flask(__name__)
app.config['SECRET_KEY'] = "sdfpAWkfs[aldpas"

@app.before_first_request
def initializeDBandModels() -> None:
    global DATABASE, AX_MODEL, RMAC_MODEL, RCNN_MODEL, EVAL_DATA
    DATABASE = Database(DATABASE_PATH)
    AX_MODEL = AlteredXception(database=DATABASE)
    RMAC_MODEL = RMACModel(AX_MODEL.model.get_layer("conv2d_3").output_shape[1:], 3, DATABASE)
    RCNN_MODEL = FasterRCNN(DATABASE)
    EVAL_DATA = load_json_data('/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/evaluation_data.json')

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

@app.route('/evaluate', methods=['GET'])
def evaluate_ui():
    return render_template('evaluate.html', eval_data=EVAL_DATA)

if __name__ == "__main__":
    app.run(debug=True)
