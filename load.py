from __future__ import division
import os

from keras import backend as K
from keras.layers import Input
from keras.models import Model

import resnet as nn#290519

import tensorflow as tf

def init():
    config = {'verbose': True, 'network': 'resnet50', 'use_horizontal_flips': False, 'use_vertical_flips': False,
              'rot_90': False, 'anchor_box_scales': [128, 256, 512],
              'anchor_box_ratios': [[1, 1], [0.7071067811865475, 1.414213562373095],
                                    [1.414213562373095, 0.7071067811865475]], 'im_size': 600,
              'img_channel_mean': [103.939, 116.779, 123.68], 'img_scaling_factor': 1.0, 'num_rois': 32,
              'rpn_stride': 16, 'balanced_classes': False, 'std_scaling': 4.0,
              'classifier_regr_std': [8.0, 8.0, 4.0, 4.0], 'rpn_min_overlap': 0.3, 'rpn_max_overlap': 0.7,
              'classifier_min_overlap': 0.1, 'classifier_max_overlap': 0.5,
              'class_mapping': {'cake': 0, 'donuts': 1, 'dosa': 2, 'bg': 3}, 'model_path': './model_frcnn.hdf5',
              'base_net_weights': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'}  # 290519
    model_path = os.path.join(os.getcwd(), 'model_frcnn.hdf5')
    num_features = 1024
    class_mapping = config['class_mapping']  # class_mapping={'cake': 0, 'donuts': 1, 'dosa': 2, 'bg': 3}
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v: k for k, v in class_mapping.items()}

    config['class_mapping'] = class_mapping  # 300519 now over-writing class_mapping label to {0: 'cake', 1: 'donuts', 2: 'dosa', 3: 'bg'}

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

    # model_classifier_only = Model([feature_map_input, roi_input], classifier)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    ##########100619
    model_rpn.load_weights(model_path, by_name=True)
    model_classifier.load_weights(model_path, by_name=True)





    print("Loaded Model Weights from disk")
    #compile and evaluate loaded model
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    graph = tf.get_default_graph()
    #graph = tf.get_default_graph()
    return model_rpn, model_classifier, config, graph#110619 #loaded_model,graph
