import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from absl import app
from absl import flags
from yolo_utils import infer_image

FLAGS = flags.FLAGS

flags.DEFINE_string(
	'weights',
	'./yolov3.weigths',
	'Path to the file which contains the weights'
)

flags.DEFINE_string(
	"config",
	"./cfg/yolov3.cfg",
	'Path to the configuration file for the YOLOv3 model.'
)

flags.DEFINE_string(
	"img_path",
	"",
	"The path to the examples file"
)

flags.DEFINE_string(
	"video_output",
	"./output.mp4",
	"The path of the output examples file"
)

flags.DEFINE_string(
	"labels",
	"/coco-labels",
	"path to the labels"
)

flags.DEFINE_float(
	"confidence",
	0.5,
	"The model will reject boundaries which has a probability less than the confidence value. default: 0.5'"
)

flags.DEFINE_float(
	"treshold",
	0.3,
	"The threshold to use when applying the Non-Max Suppresion'"
)


def main(_):
	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(cfgFile='yolov3.cfg', darknetModel='yolov3.weights')

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	if FLAGS.img_path is None:
		print('Path to examples not provided')

	elif FLAGS.img_path:
		img = cv.imread(FLAGS.img_path)

		height, width = None, None

		if width is None or height is None:
			height, width = img.shape[:2]

		img, _, _, _ = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)

		cv.imshow('examples', img)
		cv.waitKey(0)
		cv.destroyAllWindows()

	else:
		print("[ERROR] Something's not right...")


if __name__ == '__main__':
	app.run(main)



