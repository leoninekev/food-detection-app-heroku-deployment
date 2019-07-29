import flask
from flask import Flask, render_template, request

import os
import numpy as np
from keras import backend as K
#from load import *
import roi_helpers

import cv2
from scipy.misc import imsave, imread, imresize

from keras import backend as K
from keras.layers import Input
from keras.models import Model

import resnet as nn#290519

app = Flask(__name__)
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', label="empty")
        img= imread(file)

        def format_img_size(image, config):
            """ formats the image size based on config """
            img_min_side = float(config['im_size'])
            (height, width, _) = image.shape

            if width <= height:
                ratio = img_min_side / width
                new_height = int(ratio * height)
                new_width = int(img_min_side)
            else:
                ratio = img_min_side / height
                new_width = int(ratio * width)
                new_height = int(img_min_side)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return image, ratio

        def format_img_channels(image, config):
            """ formats the image channels based on config """
            image = image[:, :, (2, 1, 0)]
            image = image.astype(np.float32)
            image[:, :, 0] -= config['img_channel_mean'][0]
            image[:, :, 1] -= config['img_channel_mean'][1]
            image[:, :, 2] -= config['img_channel_mean'][2]
            image /= config['img_scaling_factor']
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)
            return image

        def format_img(image, config):
            """ formats an image for model prediction based on config """
            image, ratio = format_img_size(image, config)
            image = format_img_channels(image, config)
            return image, ratio

        # Method to transform the coordinates of the bounding box to its original size
        def get_real_coordinates(ratio, x1, y1, x2, y2):
            real_x1 = int(round(x1 // ratio))
            real_y1 = int(round(y1 // ratio))
            real_x2 = int(round(x2 // ratio))
            real_y2 = int(round(y2 // ratio))
            return (real_x1, real_y1, real_x2, real_y2)
        
        config = {'verbose': True, 'network': 'resnet50', 'use_horizontal_flips': False,
                  'use_vertical_flips': False,
                  'rot_90': False, 'anchor_box_scales': [128, 256, 512],
                  'anchor_box_ratios': [[1, 1], [0.7071067811865475, 1.414213562373095],
                                        [1.414213562373095, 0.7071067811865475]], 'im_size': 600,
                  'img_channel_mean': [103.939, 116.779, 123.68], 'img_scaling_factor': 1.0, 'num_rois': 32,
                  'rpn_stride': 16, 'balanced_classes': False, 'std_scaling': 4.0,
                  'classifier_regr_std': [8.0, 8.0, 4.0, 4.0], 'rpn_min_overlap': 0.3, 'rpn_max_overlap': 0.7,
                  'classifier_min_overlap': 0.1, 'classifier_max_overlap': 0.5,
                  'class_mapping': {'cake': 0, 'donuts': 1, 'dosa': 2, 'bg': 3}, 'model_path': './model_frcnn.hdf5',
                  'base_net_weights': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'}  # 290519
        class_mapping = config['class_mapping']#class_mapping={'cake': 0, 'donuts': 1, 'dosa': 2, 'bg': 3}
        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)
        class_mapping = {v: k for k, v in class_mapping.items()}

        config['class_mapping'] = class_mapping

        ####
        model_path = os.path.join(os.getcwd(), 'model_frcnn.hdf5')
        num_features = 1024
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(config['num_rois'], 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(config['anchor_box_scales']) * len(config['anchor_box_ratios'])
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        classifier = nn.classifier(feature_map_input, roi_input, config['num_rois'], nb_classes=len(class_mapping),
                                   trainable=True)
        model_rpn = Model(img_input, rpn_layers)
        
        model_classifier = Model([feature_map_input, roi_input], classifier)

        ####100619
        model_rpn.load_weights(model_path, by_name=True)
        model_classifier.load_weights(model_path, by_name=True)

        print("Loaded Model Weights from disk")
        # compile and evaluate loaded model
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')
        #####

        X, ratio = format_img(img, config)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)
        
        R = roi_helpers.rpn_to_roi(Y1, Y2, config, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        bbox_threshold = 0.8

        for jk in range(R.shape[0] // config['num_rois'] + 1):
            ROIs = np.expand_dims(R[config['num_rois'] * jk:config['num_rois'] * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // config['num_rois']:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], config['num_rois'], curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            
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
                    tx /= config['classifier_regr_std'][0]
                    ty /= config['classifier_regr_std'][1]
                    tw /= config['classifier_regr_std'][2]
                    th /= config['classifier_regr_std'][3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([config['rpn_stride']*x, config['rpn_stride']*y, config['rpn_stride']*(x + w),config['rpn_stride']*(y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
                
        all_dets = []
        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                coord_list = list(get_real_coordinates(ratio, x1, y1, x2, y2))
                all_dets.append((key, 100 * new_probs[jk], coord_list))

        results = all_dets
        label = 'Image named: [{a}], \n yields prediction array: [{b}]'.format(a=file, b=results)
        return render_template('index.html', label=label)

if __name__ == '__main__':
    model_rpn, model_classifier, config, graph = init()    
    app.run(host='0.0.0.0', port=8001, debug=True)

    
