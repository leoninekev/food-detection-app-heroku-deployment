import flask
from flask import Flask, render_template, request

import os
import numpy as np
from keras import backend as K
from load import *
import roi_helpers

import cv2

from keras import backend as K
from keras.layers import Input
from keras.models import Model

from tensorflow.python.lib.io import file_io#020719
from keras.preprocessing.image import img_to_array#250619
from PIL import Image#250619
import io#250619

import resnet as nn#290519
import time

class b_plate(object):
    def __init__(self,image, config, model_rpn, model_classifier):
        self._image= image#140619
        self._config = config#300519
        self._model_rpn = model_rpn
        self._model_classifier = model_classifier

    def format_img_size(self):#140619
        """ formats the image size based on config """
        img_min_side = float(self._config['im_size'])  # 140519
        (height, width, _) = self._image.shape
        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(self._image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio#this img is a modified img now

    def format_img_channels(self, img):#140519
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self._config['img_channel_mean'][0]
        img[:, :, 1] -= self._config['img_channel_mean'][1]
        img[:, :, 2] -= self._config['img_channel_mean'][2]
        img /= self._config['img_scaling_factor']
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self):#140519
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size()#220519
        img = self.format_img_channels(img)
        return img, ratio


    def preprocess(self):#300519 defined explicitly
        X, ratio = self.format_img()#140619

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = self._model_rpn.predict(X)#300519
        R = roi_helpers.rpn_to_roi(Y1, Y2, self._config, K.image_dim_ordering(), overlap_thresh=0.7)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        bboxes = {}
        probs = {}
        bbox_threshold = 0.8

        class_mapping = self._config['class_mapping']#300519

        for jk in range(R.shape[0] // self._config['num_rois'] + 1):
            ROIs = np.expand_dims(R[self._config['num_rois'] * jk:self._config['num_rois'] * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            if jk == R.shape[0] // self._config['num_rois']:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self._config['num_rois'], curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self._model_classifier.predict([F, ROIs])#300519

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])

                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self._config['classifier_regr_std'][0]
                    ty /= self._config['classifier_regr_std'][1]
                    tw /= self._config['classifier_regr_std'][2]
                    th /= self._config['classifier_regr_std'][3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self._config['rpn_stride'] * x, self._config['rpn_stride'] * y, self._config['rpn_stride'] * (x + w), self._config['rpn_stride'] * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        return [bboxes, probs, ratio]#14619 added ratio

    def get_real_coordinates(self, ratio, x1, y1, x2, y2):#220519
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))
        return (real_x1, real_y1, real_x2 ,real_y2)



    def postprocess(self, bounding_boxes, probabilities, ratio):#300519 defined explicitly
        all_dets = []
        bboxes = bounding_boxes
        probs = probabilities

        for key in bboxes:
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                coord_list = list(self.get_real_coordinates(ratio, x1, y1, x2, y2))  # 220519# addded self. to call class function
                all_dets.append((key, 100 * new_probs[jk], coord_list))

        return all_dets

app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/querry-local')#300619 try try http://localhost:8001/querry-local?image-path=C:\food-detection-020719\3.jpg
def querry_with_imagepath():
    tick= time.time()
    image_path= request.args.get('image-path')
    file= open(image_path,'rb')
    img= Image.open(io.BytesIO(file.read()))#250619
    img= img_to_array(img)#250619
    img= img[...,::-1]#140619 changed RGB(happening with scipy) to BGR(recommended by cv2 locally)
    with graph.as_default():
        #creates an object from b_plat class, performing preprocessing        
        overhead = b_plate(img, config, model_rpn, model_classifier)#140619# added img #img, config, model_rpn, model_classifier)
        [bounding_boxes,  probabilities, ratio] = overhead.preprocess()#140619 img)#outputs preprocessed outputs
        results = overhead.postprocess(bounding_boxes, probabilities, ratio)
    print('Done')
    exe_time= time.time()-tick
    label = {"filePath":image_path,"predictionTuples":results,"inferencingTime":exe_time}
    return flask.jsonify(label)
@app.route('/querry-curl', methods=['POST'])#300619@cmd try curl -X POST -F image=@3.jpg 'http://localhost:8001/querry-curl'
def querry_with_curl():
    tick= time.time()   

    if request.method=='POST':
        if request.files.get("image"):
            file = request.files["image"]

            filename= str(file)[:-17][15:]
            filepath= os.path.join(os.getcwd(),filename)
            
            img= Image.open(io.BytesIO(file.read()))
            img= img_to_array(img)#250619
            img= img[...,::-1]
            global graph#270619
            with graph.as_default():
                overhead = b_plate(img, config, model_rpn, model_classifier)#140619# added img #img, config, model_rpn, model_classifier)
                [bounding_boxes,  probabilities, ratio] = overhead.preprocess()#140619 img)#outputs preprocessed outputs
                results = overhead.postprocess(bounding_boxes, probabilities, ratio)
            exe_time= time.time()-tick
            label = {'fileName': filename,'filePath':filepath,"predictionTuples":results,"inferencingTime":exe_time}
        return flask.jsonify(label)

@app.route('/querry-gcs')#020719 try /querry-gcs?image=gs://train_on_gcloud/flask_experiments/3.jpg
def querry_with_gcs_url():
    tick= time.time()
    image_url= request.args.get('image')
    file = file_io.FileIO(image_url, mode='rb')
    
    img= Image.open(io.BytesIO(file.read()))#250619
    img= img_to_array(img)#250619
    img= img[...,::-1]#140619 changed RGB(happening with scipy) to BGR(recommended by cv2 locally)
    with graph.as_default():
        #creates an object from b_plat class, performing preprocessing        
        overhead = b_plate(img, config, model_rpn, model_classifier)#140619# added img #img, config, model_rpn, model_classifier)
        [bounding_boxes,  probabilities, ratio] = overhead.preprocess()#140619 img)#outputs preprocessed outputs
        results = overhead.postprocess(bounding_boxes, probabilities, ratio)
    print('Done')
    exe_time= time.time()-tick
    label = {"filePath":image_url,"predictionTuples":results,"inferencingTime":exe_time}
    return flask.jsonify(label)

@app.route('/b64-json-querry', methods=['POST'])#300619 @cmd try curl -X POST -F image=@{'img_data':'ac2a1ca5scab64'}.jpg 'http://localhost:8001/b64-json-querry'
def querry_with_curl_json():
    tick= time.time()
    data= {}
    
    if request.method=='POST':
        try:
            data= request.get_json()['img_data']#this is to get jsonified base64 encoding of image
        except:
            return jsonify(status_code='400', msg= 'Bad Request')

        data= base64.b64decode(data)#converts b64 incoming image to  binary data

        img= Image.open(io.BytesIO(data))
        img= img_to_array(img)#250619
        img= img[...,::-1]#140619 changed RGB(happening with scipy) to BGR(recommended by cv2 locally)

        global graph#270619
    
        with graph.as_default():
            #creates an object from b_plat class, performing preprocessing
            overhead = b_plate(img, config, model_rpn, model_classifier)#140619# added img #img, config, model_rpn, model_classifier)
            [bounding_boxes,  probabilities, ratio] = overhead.preprocess()#140619 img)#outputs preprocessed outputs
            results = overhead.postprocess(bounding_boxes, probabilities, ratio)
        print('Done')
        exe_time= time.time()-tick
        label = {"filePath":str(request.files["image"]),"predictionTuples":results,"inferencingTime":exe_time}
    return flask.jsonify(label)

@app.route('/predict_button', methods=['POST'])
def make_prediction_button():
    tick= time.time()    
    if request.method=='POST':
        file = request.files['image']
        
        filename= str(file)[:-17][15:]
        filepath= os.path.join(os.getcwd(),filename)
        
        if not file:
            return render_template('index.html', label="empty")
        
        img= Image.open(io.BytesIO(file.read()))#250619
        img= img_to_array(img)#250619
        img= img[...,::-1]#140619 changed RGB(happening with scipy) to BGR(recommended by cv2 locally)

        if not file:
            return render_template('index.html', label="empty")
        with graph.as_default():
        #creates an object from b_plat class, performing preprocessing
            overhead = b_plate(img, config, model_rpn, model_classifier)#140619# added img #img, config, model_rpn, model_classifier)
            [bounding_boxes, probabilities, ratio] = overhead.preprocess()#140619 img)#outputs preprocessed outputs
            results = overhead.postprocess(bounding_boxes, probabilities, ratio)
        print('Done')
        exe_time= time.time()-tick
        label = {'fileName': filename,'filePath':filepath,'predictionTuples':results,'inferencingTime':exe_time}
        return render_template('index.html', label=label)


if __name__ == '__main__':
    model_rpn, model_classifier, config, graph = init()    
    app.run(host='0.0.0.0', port=8001, debug=True)

    
