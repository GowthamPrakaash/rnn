#This file allows to perform Emotion detection on frames grabbed from the webcam using OpenCV-Python

import cv2
import sys
from src.fermodel import FERModel
import json
# from keras.models import load_model
from keras.models import model_from_json

from scipy import misc
import numpy as np

from pkg_resources import resource_filename

fontFace = cv2.FONT_HERSHEY_SIMPLEX;
fontScale = 1;
thickness = 2;

#Specify the camera which you want to use. The default argument is '0'
video_capture = cv2.VideoCapture(0)
#Capturing a smaller image fçor speed purposes
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
video_capture.set(cv2.CAP_PROP_FPS, 15)

#Can choose other target emotions from the emotion subset defined in fermodel.py in src directory. The function
# defined as `def _check_emotion_set_is_supported(self):`
target_emotions = ['calm', 'anger', 'happiness']
# model = FERModel(target_emotions, verbose=True)
# model_file = 'conv_lstm_model.json'
emotion_map_file = 'conv_lstm_emotion_map.json'
emotion_map = json.loads(open("output/" + emotion_map_file).read())
# model = load_model("output/" + model_file)


### newly added way to load a model
json_file = open('output/conv_lstm_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("output/conv_lstm_weights.h5")

### self declarations
target_dimensions = (48, 48)
channels = 1

while True:
	#Capture frame-by-frame
	ret, frame = video_capture.read()
	#Save the captured frame on disk
	file = 'dataset/image_data/image.jpg'
	cv2.imwrite(file, frame)

	### read each frame
	image = misc.imread(file)
	gray_image = image
	if len(image.shape) > 2:
		gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(gray_image, target_dimensions, interpolation=cv2.INTER_LINEAR)
	print(list(target_dimensions)+[channels])
	final_image = np.array([np.array([resized_image]).reshape(list(target_dimensions)+[channels])])
	prediction = loaded_model.predict(final_image)

	# frameString = loaded_model.predict(file)
	frameString = prediction

	#Display emotion
	retval, baseline = cv2.getTextSize(frameString, fontFace, fontScale, thickness)
	cv2.rectangle(frame, (0, 0 ), (20 + retval[0], 50 ), (0,0,0), -1 )
	cv2.putText(frame, frameString, (10, 35), fontFace, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)
	cv2.imshow('Video', frame)
	cv2.waitKey(1)

	#Press Esc to exit the window
	if cv2.waitKey(1) & 0xFF == 27:
		break
#Closes all windows
cv2.destroyAllWindows()


def predict(self, image_file):
	"""
	Predicts discrete emotion for given image.

	:param images: image file (jpg or png format)
	"""
	image = misc.imread(image_file)
	gray_image = image
	if len(image.shape) > 2:
		gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
	resized_image = cv2.resize(gray_image, self.target_dimensions, interpolation=cv2.INTER_LINEAR)
	final_image = np.array([np.array([resized_image]).reshape(list(self.target_dimensions)+[self.channels])])
	prediction = self.model.predict(final_image)
	# Return the dominant expression
	# dominant_expression = self._print_prediction(prediction[0])
	# return dominant_expression
	return prediction