import glob #glob module is used to retrieve files/pathnames matching a specified pattern
import os   #OS module in Python provides functions for interacting with the operating system
import sys  #sys module provides functions and variables which are used to manipulate different parts of the Python Runtime Environment
import random
import time
import numpy as np
import cv2
import argparse
import math
import queue
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
#import IPython

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#from IPython.display import display  # we wrote other function 'display_img' for this



try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg'%(
				sys.version_info.major,
				sys.version_info.minor,
				'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def display_img(image):
    cv2.imshow('img',image)
    cv2.waitKey(0)

# To load a model, we need a base url

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

# List of the strings that is used to add correct label for each box.
# We need the labelmap file now..

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)


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
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = image
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  #display(Image.fromarray(image_np))
  #display_img(image_np)
  return image_np


def processImage(image):
	image = np.array(image.raw_data)
	img = image.reshape((600,800,4))
	img = img[:,:,:3]

	img = show_inference(detection_model , img)



	cv2.imshow('img',img)
	cv2.waitKey(100)

def get_speed(vehicle):
	vel = vehicle.get_velocity()
	return 3.6*math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


class VehiclePIDController():

	def __init__(self, vehicle , args_lateral,args_longitudinal,max_throttle=0.75, max_break=0.3, max_steering=0.8):
		self.max_break = max_break
		self.max_steering = max_steering
		self.max_throttle = max_throttle

		self.vehicle = vehicle
		self.world = vehicle.get_world()
		self.past_steering = self.vehicle.get_control().steer
		self.long_controller = PIDLongitudinalControl(self.vehicle, **args_longitudinal)
		self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)

	def run_step(self,target_speed, waypoint):

		acceleration = self.long_controller.run_step(target_speed)
		current_steering = self.lat_controller.run_step(waypoint)
		control = carla.VehicleControl()

		if acceleration>=0.0:
			control.throttle = min(abs(acceleration),self.max_break)
			control.brake = 0.0
		else:
			control.throttle = 0.0
			control.brake = min(abs(acceleration),self.max_break)

		if current_steering > self.past_steering+0.1:
			current_steering = self.past_steering+0.1

		elif current_steering<self.past_steering-0.1:
			current_steering = self.past_steering-0.1
		if current_steering>=0:
			steering = min(self.max_steering , current_steering)
		else:
			steering = max(-self.max_steering , current_steering)

		control.steer = steering
		control.hand_brake = False
		control.manual_gear_shift = False
		self.past_steering = steering

		return control



class PIDLongitudinalControl():

	def __init__(self, vehicle, K_P=1.0 , K_D=0.0, K_I = 0.0 , dt=0.03):

		self.vehicle = vehicle
		self.K_D=K_D
		self.K_P=K_P
		self.K_I=K_I

		self.dt = dt
		self.errorBuffer = queue.deque(maxlen=10)

	def pid_cotroller(self, target_speed,current_speed):

		error = target_speed-current_speed

		self.errorBuffer.append(error)

		if len(self.errorBuffer)>=2:
			de = (self.errorBuffer[-1]-self.errorBuffer[-2])/self.dt

			ie = sum(self.errorBuffer)*self.dt

		else:
			de=0.0
			ie = 0.0

		return np.clip(self.K_P*error+self.K_D*de+self.K_I*ie , -1.0,1.0)


	def run_step(self,target_speed):
		current_speed = get_speed(self.vehicle)
		return self.pid_cotroller(target_speed,current_speed)

	
class PIDLateralControl():

	def __init__(self, vehicle, K_P=1.0 , K_D=0.0, K_I = 0.0 , dt=0.03):

		self.vehicle = vehicle
		self.K_D=K_D
		self.K_P=K_P
		self.K_I=K_I
		self.dt = dt
		self.errorBuffer = queue.deque(maxlen=10)

	def run_step(self,waypoint):

		return self.pid_controller(waypoint,self.vehicle.get_transform())

	
	def pid_controller(self,waypoint , vehicle_transform):

		v_begin = vehicle_transform.location
		v_end = v_begin+carla.Location(x = math.cos(math.radians(vehicle_transform.rotation.yaw)),y = math.sin(math.radians(vehicle_transform.rotation.yaw)))
		v_vec = np.array([v_end.x - v_begin.x , v_end.y - v_begin.y,0.0])

		w_vec = np.array([waypoint.transform.location.x - v_begin.x,waypoint.transform.location.y - v_begin.y,0.0])

		dot = math.acos(np.clip(np.dot(w_vec,v_vec)/np.linalg.norm(w_vec)*np.linalg.norm(v_vec),-1.0,1.0))

		cross = np.cross(v_vec,w_vec)
		if cross[2]<0:
			dot*=-1

		self.errorBuffer.append(dot)

		if len(self.errorBuffer)>=2:
			de = (self.errorBuffer[-1] - self.errorBuffer[-2])/self.dt
			ie = sum(self.errorBuffer)*self.dt

		else:
			de = 0.0
			ie = 0.0

		return np.clip((self.K_P*dot)+(self.K_I*ie)+(self.K_D*de),-1.0,1.0)






def main():
	actor_list=[]
	try:
		client = carla.Client('localhost',2000)
		client.set_timeout(10.0)
		world = client.get_world()
		map = world.get_map()

		blueprint_library = world.get_blueprint_library()
		vehicle_bp = blueprint_library.filter('cybertruck')[0]
		spawnpoint = carla.Transform(carla.Location(x=50,y=2.4,z=15),carla.Rotation(pitch=0,yaw=90,roll = 0))
		vehicle = world.spawn_actor(vehicle_bp,spawnpoint)
		actor_list.append(vehicle)
		control_vehicle = VehiclePIDController(vehicle , args_lateral={'K_P':1 , 'K_D':0.0,'K_I':0.0} , args_longitudinal={'K_P':1 , 'K_D':0.0,'K_I':0.0})


		camera_bp = blueprint_library.find('sensor.camera.rgb')
		camera_bp.set_attribute('image_size_x','800')
		camera_bp.set_attribute('image_size_y','600')
		camera_bp.set_attribute('fov','90')
		camera_transform = carla.Transform(carla.Location(x=1.5,z=2.4))
		camera = world.spawn_actor(camera_bp,camera_transform,attach_to=vehicle)
		camera.listen(lambda image : processImage(image))

		while True:
			waypoints = world.get_map().get_waypoint(vehicle.get_location())
			waypoint = np.random.choice(waypoints.next(0.3))
			control_signal = control_vehicle.run_step(3,waypoint)
			vehicle.apply_control(control_signal)


	finally:
		client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == '__main__':
	main()
