from flask import Flask, render_template, request
import os
import cv2
from PIL import Image
from infer import infer_one

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', contents=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        image = Image.open(uploaded_file)
        image.save('./static/this.png')
        #call infer.py here
        result, conf_score = infer_one(imgpath='./static/this.png')
        ##############################
        contents = '_'.join(uploaded_file.filename.split('_')[2:3]) + ' is '
        contents += 'True' if result else 'False'
        contents += ' with ' + str(conf_score) + f'% confidence'
        return render_template('index.html', contents=contents)
    else:
        return render_template('index.html', contents=None)
    

if __name__ == '__main__':
    app.run(debug=False)

