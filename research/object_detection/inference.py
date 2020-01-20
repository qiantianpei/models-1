import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from PIL import Image
import json
import time
import pickle

sys.path.append('/home/tianpei/workspace')

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from keras_retinanet.utils.gpu import setup_gpu

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile




def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'

  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.compat.v2.saved_model.load(str(model_dir), None) #tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

print(category_index)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path('test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)


model_name = 'faster_rcnn_nas_coco_2018_01_28'
detection_model = load_model(model_name)

remap = {
    3: 0, 
    6: 1, 
    8: 2,
    4: 3,
    1: 4,
    2: 5   
}




# GRAPH_PB_PATH = '/home/tianpei/Downloads/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb'
# with tf.Session() as sess:
#    print("load graph")
#    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
#        graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())
#    sess.graph.as_default()
#    tf.import_graph_def(graph_def, name='')
#    graph_nodes=[n for n in graph_def.node]
#    names = []
#    for t in graph_nodes:
#       names.append(t.name)
#    print(names)

print(detection_model.inputs)

print(detection_model.output_dtypes)
print(detection_model.output_shapes)

# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
#   with tf.Session() as sess:
#     init = tf.compat.v1.global_variables_initializer()
#     sess.run(init)
#     num_detections = sess.run(output_dict.pop('num_detections'))
#   print(num_detections)
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  return output_dict




def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  boxes, scores, labels = output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes']
  labels = np.array([remap[label] if label in remap else label for label in labels])
  return boxes, scores, labels

  # vis_util.visualize_boxes_and_labels_on_image_array(
  #     image_np,
  #     output_dict['detection_boxes'],
  #     output_dict['detection_classes'],
  #     output_dict['detection_scores'],
  #     category_index,
  #     instance_masks=output_dict.get('detection_masks_reframed', None),
  #     use_normalized_coordinates=True,
  #     line_thickness=8)

  # Image.fromarray(image_np).show()

data_path = '/home/tianpei/workspace/data_local/us/'
test_file = 'us_test_1.1.json'
with open(data_path + test_file) as f:
    test = json.load(f) 

boxes_all = []
scores_all = []
labels_all = []


for im in test[:2]:
  start = time.time()
  boxes, scores, labels = show_inference(detection_model, data_path + im['file'])
  print(boxes.shape)
  boxes = np.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis = 1)
  print(boxes.shape)
  print("processing time: ", time.time() - start)
  boxes_all.append(boxes[np.logical_and(scores > 0.5, labels <= 5)])
  scores_all.append(scores[np.logical_and(scores > 0.5, labels <= 5)])
  labels_all.append(labels[np.logical_and(scores > 0.5, labels <= 5)])
  

# np.save('us_test_1.1_nasnet_output.npy', [boxes_all, scores_all, labels_all], protocol=2)

pickle.dump([boxes_all, scores_all, labels_all], open('us_test_1.1_nasnet_output.npy', "wb" ) , protocol=2)