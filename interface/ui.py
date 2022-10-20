import tempfile
from flask import Flask, render_template, request
import sys
import os
sys.path.append('..')
from database import Database
from query import query_the_image

app = Flask(__name__)
app.config['SECRET_KEY'] = "sdfpAWkfs[aldpas"
download_dir = "/Users/lchris/Desktop/Coding/schoolprojects/comp490/COMPS/data"

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    file_url = None
    top_yt_links = None
    db = Database(download_dir)
    if request.method == 'POST':
        imagefile = request.files['imagefile'].read()
        im_request_file = request.files.get('imagefile', '')
        ext = '.' + im_request_file.filename.split('.')[1]
        with tempfile.NamedTemporaryFile(suffix=ext, mode='w+b', dir="./static/images", delete=False) as temp_img_file:
            temp_img_file.write(imagefile)
            file_url = os.path.relpath(temp_img_file.name)
            top_yt_links = query_the_image(temp_img_file.name, db, 10)
    return render_template('index.html', file_url=file_url, yt_links=top_yt_links)
    
if __name__ == "__main__":
    app.run(debug=True)
