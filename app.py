#. venv/bin/activate
#export FLASK_ENV=development
#flask run
from xml.sax.handler import feature_validation
from flask import request, redirect, render_template, url_for
from flask import Flask
from werkzeug.utils import secure_filename
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from scipy import stats
import cv2

import os
import cv2
import numpy as np

app = Flask(__name__)
if __name__ == 'main':
    app.run(host='0.0.0.0')
app.config["IMAGE_UPLOADS"] = "static/uploads/"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024
svm_path = "finalized_model.sav"
rf_path = "model/finalized_model_rfc.sav"

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
                result = model(os.path.join(app.config["IMAGE_UPLOADS"], filename),svm_path)
                print("Result ", result[0])
                finalRes1 = ""
                if result[0] == 1:
                    finalRes1 = "Pneumonia Detected"
                else:
                    finalRes1 = "No Pneumonia"
                print(finalRes1)
                
                
                finalRes2 = ""
                """
                result = model(os.path.join(app.config["IMAGE_UPLOADS"], filename),rf_path)
                print("Result ", result[0])
                
                if result[0] == 1:
                    finalRes2 = "Pneumonia Detected"
                else:
                    finalRes2 = "No Pneumonia"
                print(finalRes2)
                """
                return render_template("upload_image.html", finalRes1 = finalRes1, finalRes2 = finalRes2)

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

def image_convert2(path,fs):
    images = []
    labels = []
    normal_limit=1
    p_limit=0
    img_size=256

    img = cv2.imread(path) 
    if img is not None:
        img  = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)   
        img = cv2.resize(img,(img_size, img_size))
        images.append(img)
    images,labels=np.array(images),np.array(labels)
    print(images.shape)
    #print(labels)
    
    #Obtaining Variance of images
    kernel = np.ones((3,3),np.uint8)
    #print(kernel)
    var_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        x, bins = np.histogram(image,bins=255, density=False)
        var_vector[i] = np.var(x)
        i=i+1
    #print(var_vector[6])
    
    #Obtaining lbp of images
    from skimage.feature import multiblock_lbp
    lbp_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        lbp = multiblock_lbp(image, 0,0,28,28)
        lbp_vector[i] = lbp
        i=i+1
    print("LBP Vector shape", lbp_vector.shape)
    
    #Obtaining Mean of images
    mean_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        x, bins = np.histogram(image,bins=255, density=False)
        mean_vector[i] = np.mean(x)
        i=i+1
    #print(mean_vector[6])
    
    #Obtaining Standard Deviation of images
    std_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        x, bins = np.histogram(image,bins=255, density=False)
        std_vector[i] = np.std(x)
        i=i+1
    #print(std_vector[6])
    
    skew_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        x, bins = np.histogram(image,bins=255, density=False)
        skew_vector[i] = stats.skew(x)
        i=i+1
    #print(skew_vector[6])
    
    #Obtaining Kurtosis of images
    kurto_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        x, bins = np.histogram(image,bins=255, density=False)
        kurto_vector[i] = stats.kurtosis(x)
        i=i+1
    #print(kurto_vector[6])
    #Obtaining Entropy of images
    entropy_vector = np.empty((normal_limit+p_limit,1))
    i = 0
    for image in images:
        x, bins = np.histogram(image,bins=255, density=False)
        entropy_vector[i] = stats.entropy(x)
        i=i+1
    #print(entropy_vector[6])
    print(np.shape(entropy_vector))
    
    #Applying Canny edge detection
    canny_vector = np.empty((normal_limit+p_limit,img_size*img_size))
    i = 0
    for image in images:
        canny = cv2.Canny(image,40,200)
        canny_vector[i] = np.array(canny.flatten())
        i=i+1
    print(np.shape(canny_vector))
    
    #Applying Sobel X
    sobelX_vector = np.empty((normal_limit+p_limit,img_size*img_size))
    i = 0
    for image in images:
        sobelX = cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5)
        sobelX_vector[i] = np.array(sobelX.flatten())
        i=i+1
    #Applying Sobel Y
    sobelY_vector = np.empty((normal_limit+p_limit,img_size*img_size))
    i = 0
    for image in images:
        sobelY = cv2.Sobel(image,cv2.CV_8UC1,0,1,ksize=5)
        sobelY_vector[i] = np.array(sobelY.flatten())
        i=i+1

    feature_vector = np.empty((normal_limit+p_limit,0))
    print("NL", normal_limit)
    print("PL", p_limit)
    #feature_vector=np.append(feature_vector,mean_vector,axis=1)
    feature_vector=np.append(feature_vector,lbp_vector,axis=1)
    feature_vector=np.append(feature_vector,var_vector,axis=1)
    feature_vector=np.append(feature_vector,std_vector,axis=1)
    feature_vector=np.append(feature_vector,skew_vector,axis=1)
    feature_vector=np.append(feature_vector,kurto_vector,axis=1)
    feature_vector=np.append(feature_vector,entropy_vector,axis=1)
    feature_vector=np.append(feature_vector,canny_vector,axis=1)
    feature_vector=np.append(feature_vector,sobelX_vector,axis=1)
    feature_vector=np.append(feature_vector,sobelY_vector,axis=1)
    #feature_vector=np.append(feature_vector,hog_features,axis=1)
    print("FV",np.shape(feature_vector))
    
    fs = np.append(fs,feature_vector,axis=1)
    
    print((fs))
    return fs



def model(path, model_path):
    print("PATH ", path)
    fs = np.empty((1,0))
    fs2 = image_convert2(path,fs)
    #fs2= np.append(fs,feature_vector[1],axis=1)
    print("FS after",np.shape(fs2) )
    #print("Feature vecotr shape [1]: ", np.shape([feature_vector[1]]))
    #fs2= np.append(fs2,[feature_vector[1]],axis=0)
    print("FS after after",np.shape(fs2) )
    #sc = StandardScaler()
    #fs2 = sc.fit_transform(fs2)
    print(fs2)
    loaded_model = pickle.load(open(model_path, 'rb'))
    #loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
    y_prob = loaded_model.predict(fs2)
    print(y_prob, "  *** OUTPUTZZ ****")
    return y_prob
"""
#image_convert2("static/uploads/Screen_Shot_2022-04-03_at_12.27.29_PM.png")
fs = np.empty((1,0))
fs2 = image_convert2("static/uploads/normal3.jpeg",fs)
#fs2= np.append(fs,feature_vector[1],axis=1)
print("FS after",np.shape(fs2) )
#print("Feature vecotr shape [1]: ", np.shape([feature_vector[1]]))
#fs2= np.append(fs2,[feature_vector[1]],axis=0)
print("FS after after",np.shape(fs2) )
#sc = StandardScaler()
#fs2 = sc.fit_transform(fs2)
print(fs2)
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
y_prob = loaded_model.predict(fs2)
print(y_prob, "  *** OUTPUT ****")
"""