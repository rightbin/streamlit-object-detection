import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from PIL import Image,ImageFilter,ImageEnhance
import h5py
import tensorflow.keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle 
import joblib
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing import image
import tensorflow as tf
import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np 

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

PATH_TO_LABELS = './mscoco_label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

detection_model = load_model(model_name)

detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes


def run_ml_app():
    st.subheader('object detection')
    st.write('이제 실제로 한번 object detection을 경험해볼까요?')
    st.write('사람,물체 등이 잘 나타나있는 사진을 한번 올려보세요!')
    
    image_file = None
    image_file = st.file_uploader("Upload Image", type=["png","jpg",'jpeg'])
    
    if image_file is None :
        st.write("사진을 업로드 해주세요.")

    else :

        def run_inference_for_single_image(model, image):
            image = np.asarray(image)
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis,...]

            # Run inference
            model_fn = model.signatures['serving_default']
            output_dict = model_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.
            num_detections = int(output_dict.pop('num_detections'))
            output_dict = {key:value[0, :num_detections].numpy() 
                            for key,value in output_dict.items()}
            output_dict['num_detections'] = num_detections

            # detection_classes should be ints.
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
            
            # Handle models with masks:
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = tf.convert_to_tensor(output_dict['detection_masks'], dtype=tf.float32)
                output_dict['detection_boxes'] = tf.convert_to_tensor(output_dict['detection_boxes'], dtype=tf.float32)
                # Reframe the the bbox mask to the image size.
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        output_dict['detection_masks'], output_dict['detection_boxes'],
                        image.shape[0], image.shape[1])  
                detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                                tf.uint8)
                output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
                
            return output_dict

        def show_inference(model):
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = np.array(Image.open(image_file))
            # Actual detection.
            output_dict = run_inference_for_single_image(model, image_np)
            # Visualization of the results of a detection.
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.array(output_dict['detection_boxes']),
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed',None),
                use_normalized_coordinates=True,
                line_thickness=8)

            return image_np

        image_np = show_inference(detection_model)
        # print(type(image_np))
        pil_image=Image.fromarray(image_np)
                
        st.image(pil_image)