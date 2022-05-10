#. venv/bin/activate
#export FLASK_ENV=development
#flask run
from flask import request, redirect, render_template, url_for
from flask import Flask
from werkzeug.utils import secure_filename


import os
import numpy as np

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/uploads/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    print(request.url)
    if request.method == "POST":
        print("1")
        if request.files:
            print("2", request.files)    
            
            image = request.files["image"]
            print("4")
            if image.filename == "":
                print("No filename")
                return redirect(request.url)
            print("5")
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                print("Image saved")

                return redirect(request.url)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)
            print("6")
    return render_template("upload_image.html")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)