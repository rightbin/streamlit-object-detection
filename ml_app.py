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

# from object_detection.utils import ops as utils_ops
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

PATH_TO_LABELS = """item {
  name: "/m/01g317"
  id: 1
  display_name: "person"
}
item {
  name: "/m/0199g"
  id: 2
  display_name: "bicycle"
}
item {
  name: "/m/0k4j"
  id: 3
  display_name: "car"
}
item {
  name: "/m/04_sv"
  id: 4
  display_name: "motorcycle"
}
item {
  name: "/m/05czz6l"
  id: 5
  display_name: "airplane"
}
item {
  name: "/m/01bjv"
  id: 6
  display_name: "bus"
}
item {
  name: "/m/07jdr"
  id: 7
  display_name: "train"
}
item {
  name: "/m/07r04"
  id: 8
  display_name: "truck"
}
item {
  name: "/m/019jd"
  id: 9
  display_name: "boat"
}
item {
  name: "/m/015qff"
  id: 10
  display_name: "traffic light"
}
item {
  name: "/m/01pns0"
  id: 11
  display_name: "fire hydrant"
}
item {
  name: "/m/02pv19"
  id: 13
  display_name: "stop sign"
}
item {
  name: "/m/015qbp"
  id: 14
  display_name: "parking meter"
}
item {
  name: "/m/0cvnqh"
  id: 15
  display_name: "bench"
}
item {
  name: "/m/015p6"
  id: 16
  display_name: "bird"
}
item {
  name: "/m/01yrx"
  id: 17
  display_name: "cat"
}
item {
  name: "/m/0bt9lr"
  id: 18
  display_name: "dog"
}
item {
  name: "/m/03k3r"
  id: 19
  display_name: "horse"
}
item {
  name: "/m/07bgp"
  id: 20
  display_name: "sheep"
}
item {
  name: "/m/01xq0k1"
  id: 21
  display_name: "cow"
}
item {
  name: "/m/0bwd_0j"
  id: 22
  display_name: "elephant"
}
item {
  name: "/m/01dws"
  id: 23
  display_name: "bear"
}
item {
  name: "/m/0898b"
  id: 24
  display_name: "zebra"
}
item {
  name: "/m/03bk1"
  id: 25
  display_name: "giraffe"
}
item {
  name: "/m/01940j"
  id: 27
  display_name: "backpack"
}
item {
  name: "/m/0hnnb"
  id: 28
  display_name: "umbrella"
}
item {
  name: "/m/080hkjn"
  id: 31
  display_name: "handbag"
}
item {
  name: "/m/01rkbr"
  id: 32
  display_name: "tie"
}
item {
  name: "/m/01s55n"
  id: 33
  display_name: "suitcase"
}
item {
  name: "/m/02wmf"
  id: 34
  display_name: "frisbee"
}
item {
  name: "/m/071p9"
  id: 35
  display_name: "skis"
}
item {
  name: "/m/06__v"
  id: 36
  display_name: "snowboard"
}
item {
  name: "/m/018xm"
  id: 37
  display_name: "sports ball"
}
item {
  name: "/m/02zt3"
  id: 38
  display_name: "kite"
}
item {
  name: "/m/03g8mr"
  id: 39
  display_name: "baseball bat"
}
item {
  name: "/m/03grzl"
  id: 40
  display_name: "baseball glove"
}
item {
  name: "/m/06_fw"
  id: 41
  display_name: "skateboard"
}
item {
  name: "/m/019w40"
  id: 42
  display_name: "surfboard"
}
item {
  name: "/m/0dv9c"
  id: 43
  display_name: "tennis racket"
}
item {
  name: "/m/04dr76w"
  id: 44
  display_name: "bottle"
}
item {
  name: "/m/09tvcd"
  id: 46
  display_name: "wine glass"
}
item {
  name: "/m/08gqpm"
  id: 47
  display_name: "cup"
}
item {
  name: "/m/0dt3t"
  id: 48
  display_name: "fork"
}
item {
  name: "/m/04ctx"
  id: 49
  display_name: "knife"
}
item {
  name: "/m/0cmx8"
  id: 50
  display_name: "spoon"
}
item {
  name: "/m/04kkgm"
  id: 51
  display_name: "bowl"
}
item {
  name: "/m/09qck"
  id: 52
  display_name: "banana"
}
item {
  name: "/m/014j1m"
  id: 53
  display_name: "apple"
}
item {
  name: "/m/0l515"
  id: 54
  display_name: "sandwich"
}
item {
  name: "/m/0cyhj_"
  id: 55
  display_name: "orange"
}
item {
  name: "/m/0hkxq"
  id: 56
  display_name: "broccoli"
}
item {
  name: "/m/0fj52s"
  id: 57
  display_name: "carrot"
}
item {
  name: "/m/01b9xk"
  id: 58
  display_name: "hot dog"
}
item {
  name: "/m/0663v"
  id: 59
  display_name: "pizza"
}
item {
  name: "/m/0jy4k"
  id: 60
  display_name: "donut"
}
item {
  name: "/m/0fszt"
  id: 61
  display_name: "cake"
}
item {
  name: "/m/01mzpv"
  id: 62
  display_name: "chair"
}
item {
  name: "/m/02crq1"
  id: 63
  display_name: "couch"
}
item {
  name: "/m/03fp41"
  id: 64
  display_name: "potted plant"
}
item {
  name: "/m/03ssj5"
  id: 65
  display_name: "bed"
}
item {
  name: "/m/04bcr3"
  id: 67
  display_name: "dining table"
}
item {
  name: "/m/09g1w"
  id: 70
  display_name: "toilet"
}
item {
  name: "/m/07c52"
  id: 72
  display_name: "tv"
}
item {
  name: "/m/01c648"
  id: 73
  display_name: "laptop"
}
item {
  name: "/m/020lf"
  id: 74
  display_name: "mouse"
}
item {
  name: "/m/0qjjc"
  id: 75
  display_name: "remote"
}
item {
  name: "/m/01m2v"
  id: 76
  display_name: "keyboard"
}
item {
  name: "/m/050k8"
  id: 77
  display_name: "cell phone"
}
item {
  name: "/m/0fx9l"
  id: 78
  display_name: "microwave"
}
item {
  name: "/m/029bxz"
  id: 79
  display_name: "oven"
}
item {
  name: "/m/01k6s3"
  id: 80
  display_name: "toaster"
}
item {
  name: "/m/0130jx"
  id: 81
  display_name: "sink"
}
item {
  name: "/m/040b_t"
  id: 82
  display_name: "refrigerator"
}
item {
  name: "/m/0bt_c3"
  id: 84
  display_name: "book"
}
item {
  name: "/m/01x3z"
  id: 85
  display_name: "clock"
}
item {
  name: "/m/02s195"
  id: 86
  display_name: "vase"
}
item {
  name: "/m/01lsmm"
  id: 87
  display_name: "scissors"
}
item {
  name: "/m/0kmg4"
  id: 88
  display_name: "teddy bear"
}
item {
  name: "/m/03wvsk"
  id: 89
  display_name: "hair drier"
}
item {
  name: "/m/012xff"
  id: 90
  display_name: "toothbrush"
}"""


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)



model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

detection_model = load_model(model_name)

detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes


def run_ml_app():
    st.subheader('object detection')
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
        print(type(image_np))
        pil_image=Image.fromarray(image_np)
                
        st.image(pil_image)