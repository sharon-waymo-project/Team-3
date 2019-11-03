import os
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
from os import listdir
from os.path import isfile, join

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from shapely.geometry import Polygon, LineString
import random
import sys
import csv

# TODO: Change this to your own setting
os.environ['PYTHONPATH']='/env/python:~/github/waymo-open-dataset'
m=imp.find_module('waymo_open_dataset', ['.'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

TYPE_VEHICLE = 1
FRONT = 1
FPS = 10
VERIFY_THRESHOLD = 0.05
FRONT_CAR_TOLERANCE = 0.5

#Eager execution is enabled by default in TF 2.0 Therefore we need not enable it explicitly.
#tf.enable_eager_execution()

total_frames = 0
skipped_frames = 0

"""
features:
[vx, vy, vz, dx, dy, vfx, vfy, vfz, afx, afy, afz, num_v_labels]

labels:
[ax, ay, az]
"""

def write_to_csv(filename, feats, labels):
  data = np.hstack((feats, labels))
  with open(filename,'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['vx', 'vy', 'vz', 'dx', 'dy', 'vfx', 'vfy', 'vfz', 'afx', 'afy', 'afz', 'ax', 'ay', 'az', 'num_v_labels'])
    w.writerows(data)


def intersects(label, tolerance):
  # Starting from the upper-left corner, clockwise
  bounding_box = Polygon([
                          (label.box.center_x + 0.5 * label.box.length, label.box.center_y + 0.5 * label.box.width),
                          (label.box.center_x + 0.5 * label.box.length, label.box.center_y - 0.5 * label.box.width),
                          (label.box.center_x - 0.5 * label.box.length, label.box.center_y - 0.5 * label.box.width),
                          (label.box.center_x - 0.5 * label.box.length, label.box.center_y + 0.5 * label.box.width)
                          ])

  tolerance_box = Polygon([
                           (label.box.center_x, tolerance), (label.box.center_x, -tolerance),
                           (0, -tolerance), (0, tolerance)
                           ])

  return bounding_box.intersects(tolerance_box)



def collect_vehicle_laser_labels(frame):
  return [vehicle_laser_label for vehicle_laser_label in frame.laser_labels if vehicle_laser_label.type == TYPE_VEHICLE]


def get_vehicle_pose(frame):
    # get the pose of the front view
    front_image = frame.images[FRONT]
    pose = [t for t in front_image.pose.transform]
    return np.asarray(pose).reshape((4,4))


def get_current_car_velocity_wrt_GF(frame):
  image = frame.images[FRONT]
  return np.asarray([image.velocity.v_x, image.velocity.v_y, image.velocity.v_z])


def get_front_car_velocity_wrt_GF(front_car_label, vehicle_pose, v_cur_GF):
  v_front_VF = np.asarray([front_car_label.metadata.speed_x, front_car_label.metadata.speed_y, 0])
  _v_front_VF = np.hstack((v_front_VF, [0])) # padded 0 for matrix multiplication
  return np.matmul(vehicle_pose, _v_front_VF)[:3] - v_cur_GF


def get_relative_distance(front_car_label):
  return np.asarray([front_car_label.box.center_x, front_car_label.box.center_y])


def get_current_car_accel_GF_per_frame(dt, v_cur_GF, v_cur_GF_prev):
  return cal_aceleration(v_cur_GF, v_cur_GF_prev, dt) if v_cur_GF_prev is not None else np.asarray([0, 0, 0])


def get_front_car_GF_features_per_frame(dt, frame, vehicle_pose, front_car_label,
                                        v_cur_GF, v_front_GF_prev, verify=False):

  if verify and random.random() < VERIFY_THRESHOLD:
    verify_front_car_label(frame, front_car_label)

  relative_dist = get_relative_distance(front_car_label) # 2 * 1
  v_front_GF = get_front_car_velocity_wrt_GF(front_car_label, vehicle_pose, v_cur_GF) # 3 * 1
  a_front_GF = cal_aceleration(v_front_GF, v_front_GF_prev, dt) if v_front_GF_prev is not None else np.asarray([0, 0, 0]) # 3 * 1

  return np.hstack((relative_dist, v_front_GF, a_front_GF)), v_front_GF


def get_front_car_laser_label(labels, detection_tolerance):
  """
  Find the closest bounding box which intersects with y = 0 and its center_x is positive
  """

  front_car_label = None

  for label in labels:
    if label.box.center_x < 0:
      continue

    if intersects(label, detection_tolerance):
      if front_car_label is None or front_car_label.box.center_x > label.box.center_x:
        front_car_label = label

  return front_car_label


def cal_aceleration(v1, v2, dt):
  return (v2 - v1) / dt


def get_essentials_per_frame(dt, frame, front_car_label, v_cur_GF_prev, v_front_GF_prev):
  vehicle_pose = get_vehicle_pose(frame)
  v_cur_GF = get_current_car_velocity_wrt_GF(frame) # 3 * 1
  front_GF_feat, v_front_GF = get_front_car_GF_features_per_frame(dt, frame, vehicle_pose, front_car_label,
                                                                  v_cur_GF, v_front_GF_prev) # 8 * 1
  a_cur_GF = get_current_car_accel_GF_per_frame(dt, v_cur_GF, v_cur_GF_prev) # 3 * 1

  return np.hstack((v_cur_GF, front_GF_feat)), a_cur_GF, v_cur_GF, v_front_GF


def get_features_and_labels(frames, detection_tolerance=0.0):
  global total_frames
  global skipped_frames

  feat_set = []
  label_set = []

  dt = len(frames) * 1.0 / FPS / (len(frames) - 1)

  # init
  v_cur_GF_prev = None
  v_front_GF_prev = None

  for frame in frames:
    total_frames += 1

    # Capture the front car
    v_laser_labels = collect_vehicle_laser_labels(frame)
    front_car_label = get_front_car_laser_label(v_laser_labels, detection_tolerance)

    if front_car_label is not None:
       feats, labels, v_cur_GF_prev, v_front_GF_prev = get_essentials_per_frame(dt, frame, front_car_label, v_cur_GF_prev, v_front_GF_prev)
       feats = np.hstack((feats, len(v_laser_labels)))
    else:
        skipped_frames += 1
        print("No front car captured, will skip this frame")

        #if there is no front car
        v_cur_GF = get_current_car_velocity_wrt_GF(frame)
        vx, vy, vz = v_cur_GF
        feats = [vx, vy, vz, 0, 0, 0, 0, 0, 0, 0, 0, len(v_laser_labels)]
        ax, ay, az = [0,0,0]

        if v_cur_GF_prev is not None:
            ax, ay, az = get_current_car_accel_GF_per_frame(dt, v_cur_GF, v_cur_GF_prev)
        labels = [ax, ay, az]

        v_cur_GF_prev = v_cur_GF
        v_front_GF_prev = None

    feat_set.append(feats)
    label_set.append(labels)


  return np.asarray(feat_set), np.asarray(label_set)


if __name__ == "__main__":
    seg_path = sys.argv[1]
    result_path = sys.argv[2]
    detection_tolerance = int(sys.argv[3])

    files = [join(seg_path, f) for f in listdir(seg_path) if isfile(join(seg_path, f)) and ".tfrecord" in f]

    print("Total files: ", len(files))
    print("Detection tolerance: ", detection_tolerance)

    feat_set = None
    label_set = None

    for i in range(len(files)):
      file = files[i]
      print(file)
      dataset = tf.data.TFRecordDataset(file, compression_type='')

      # Load frames from dataset
      frames = []
      for data in dataset:
          frame = open_dataset.Frame()
          frame.ParseFromString(bytearray(data.numpy()))
          frames.append(frame)

      print("filename:", file, "Num of frames:", len(frames))

      feats, labels = get_features_and_labels(frames, detection_tolerance)

      feat_set = np.vstack((feat_set, feats)) if feat_set is not None else feats
      label_set = np.vstack((label_set, labels)) if label_set is not None else labels

      print("Progress: ", i, "/", len(files))

    write_to_csv(result_path, feat_set, label_set)
    print("Finished! Total_frames = ", total_frames, " skipped_frames = ", skipped_frames)
