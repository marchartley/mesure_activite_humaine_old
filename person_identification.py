import cv2
import numpy as np
import glob
import skeleton_tracker as st
import pyrealsense2 as rs
import face_recognition  # Must be imperatively placed AFTER import pyrealsense2
import colorsys
import time

import matplotlib.pyplot as plt

# from convert_realsense_to_real_world import *

import human_skeleton
import InfluxDB_connection
import recording

import pickle
import os.path
import sklearn.svm as svm

def create_encodings(images_filenames, save_filename = None, verbose = False):
	all_encodings = []
	if verbose:
		print("Encoding known faces")
	for i, img in enumerate(images_filenames):
		if verbose:
			print(f"{round((i / len(images_filenames)) * 100)}%")

		np_img = cv2.imread(img, 0)
		dim = np_img.shape[:2]
		try:
			encoding = face_recognition.face_encodings(face_recognition.load_image_file(img), model = 'large')
			all_encodings.append(encoding[0])
		except:
			encoding = face_recognition.face_encodings(face_recognition.load_image_file(img),
			                                           [(0, dim[1], dim[0], 0)], model = 'large')
			all_encodings.append(encoding[0])
	if verbose:
		print("100%")
	if save_filename is not None:
		os.makedirs(os.path.join(*os.path.split(save_filename)[:-1]), exist_ok = True)
		with open(save_filename, "wb+") as file:
			pickle.dump(all_encodings, file)
	return all_encodings


def get_encodings(filename, verbose = False):
	all_encodings = []
	if verbose:
		print(f"Looking for {filename}")
	with open(filename, "rb+") as file:
		all_encodings = pickle.load(file)
	if verbose:
		print(f"{len(all_encodings)} encodings found")
	return all_encodings


def recognize_person(human, frame, all_faces_encodings = None, classifier = None):
	if human.get_face_location() is None:
		return None
	face_locations = [human.get_face_location()] # Create an array with one element that are the coords of the face
	face_encodings = face_recognition.face_encodings(frame, face_locations, model = 'small')
	probabilities = []
	best_match_index = None

	if len(face_encodings) == 1: # Only one face to detect here
		encoding = face_encodings[0]
		# matches = face_recognition.compare_faces(all_faces_encodings, encoding)
		if all_faces_encodings is not None:
			probabilities = 1 - face_recognition.face_distance(all_faces_encodings, encoding)
			best_match_index = np.argmin(probabilities)
		elif classifier is not None:
			best_match_index = classifier.predict([encoding])[0]
			probabilities = classifier.predict_proba([encoding])[0]
			probabilities[best_match_index] *= 3
			# [0] * 1000
			# probabilities[best_match_index] = 1.0
			# probabilities = classifier.predict_proba([encoding])[0]
			# best_match_index = np.argmax(probabilities)
	return np.array(probabilities), best_match_index

def train_classifier_for_person_identification(images, filename_to_save_classifier = None):
	truth = []
	data = []

	# Getting data and ground truth:
	for id_person, image_collection in enumerate(images):
		for img in image_collection:
			np_img = cv2.imread(img, 0)
			dim = np_img.shape[:2]
			try:
				encoding = face_recognition.face_encodings(face_recognition.load_image_file(img), model = 'large')
				data.append(encoding[0])
			except Exception:
				encoding = face_recognition.face_encodings(face_recognition.load_image_file(img),
				                                           [(0, dim[1], dim[0], 0)], model = 'large')
				data.append(encoding[0])
			truth.append(id_person)
	# Creating a classifier and fitting it:
	classifier = svm.SVC(gamma = 'scale', probability = True)
	classifier.fit(data, truth)

	os.makedirs(os.path.join(*os.path.split(filename_to_save_classifier)[:-1]), exist_ok = True)
	if filename_to_save_classifier:
		with open(filename_to_save_classifier, "wb+") as file:
			pickle.dump(classifier, file)
	return classifier

def get_classifier_for_person_identification(filename):
	with open(filename, "rb+") as file:
		return pickle.load(file)


if __name__ == "__main__":

	images = [
		['faces/Eric-Benoit.jpg'],
		['faces/Marc-Hartley.jpg', 'faces/marc_face.jpg'],
		['faces/Alexandre-Benoit.jpg'],
		['faces/Stephane-Perrin.jpg'],
	]
	labels = [
		'Eric',
		'Marc',
		'Alexandre',
		'Stéphane',
	]

	# all_encodings = create_encodings(images, "faces/all_encodings_all_blured.encod", verbose = True)
	# all_encodings = get_encodings("faces/all_encodings.encod")
	classifier = train_classifier_for_person_identification(images, "models/svc.classif")
	# classifier = get_classifier_for_person_identification("models/svc.classif")

	align_to = rs.stream.depth
	align = rs.align(align_to)

	pipeline = rs.pipeline()
	# skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline = pipeline)
	pipeline = recording.start_playback("recordings/walking_normal.bag", pipeline = pipeline, realtime = False)
	skeleton_db2D, skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints(
		"recordings/walking_normal.bag")
	# pipeline.start() # Live mode
	joint_confidence = 0.4

	# backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history=500)
	num_frame = 0

	start_time = time.time()
	human = None
	playing = True

	current_timestamp = 0
	previous_timestamp = 0
	# automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline)
	found_person_name = ""
	proba = 0.0

	face_locations = []
	face_distances = np.array([])
	number_of_faces_detection = 0

	min_height = 1000

	mini = np.zeros((1, 1, 3))

	while True:
		if playing:
			previous_timestamp = current_timestamp
			try:
				frames = pipeline.wait_for_frames()
			except Exception as e:
				print(e)
				break
			num_frame += 1
			# frames = align.process(frames)
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()

			current_timestamp = frames.timestamp

			color_frame = np.asanyarray(color_frame.get_data()).copy()
			color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
			rgb_small_frame = color_frame.copy()  # cv2.resize(color_frame.copy(), (0, 0), fx = 1 / 4, fy = 1 / 4)

			# skeletons = skeletrack.track_skeletons(color_frame) # Live mode

			try:
				# for skeleton_id in range(len(skeletons)):   # Live mode
				for skeleton_id in range(len(skeleton_db2D)):  # InfluxDB mode
					human = human_skeleton.HumanSkeleton()
					# human.from_tracker(skeletrack, skeleton_id, depth_frame) # Live mode
					human.from_json(data2D = skeleton_db2D[skeleton_id], data3D = skeleton_db3D[skeleton_id],
					                specific_timestamp = current_timestamp / 1000)  # InfluxDB mode

					face_locations = [human.get_face_location()]
					face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model = 'small')
					best_match = -1
					distance = []

					for encoding in face_encodings:
						# Using the classifier
						found_person_name = labels[classifier.predict([encoding])[0]]
						proba = 0.0

						# Using the encoding distance
						# # matches = face_recognition.compare_faces(all_encodings, encoding)
						#
						# if len(face_distances) == 0:  # len(matches):
						# 	face_distances = face_recognition.face_distance(all_encodings, encoding)
						# else:
						# 	distances = face_recognition.face_distance(all_encodings, encoding)
						# 	if distances is not None:
						# 		face_distances += distances
						# 		number_of_faces_detection += 1
						# best_match_index = np.argmin(face_distances)
						# # if matches[best_match_index]:
						# found_person_name = labels[best_match_index]
						# proba = face_distances[best_match_index] / face_distances.sum()
						# print(1 - (face_distances / face_distances.sum()))


			except Exception as e:
				print(e)

			color_frame = cv2.putText(color_frame, str(round((1 - proba) * 100, 2)) + "%: " + found_person_name,
			                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

			top, right, bottom, left = 0, 1, 1, 0
			if len(face_locations) > 0:
				if face_locations[0] is not None:
					top, right, bottom, left = face_locations[0]
					min_height = min(min_height, abs(bottom - top))
					mini = rgb_small_frame[top:bottom, left:right]
			mini = cv2.resize(mini, (800, 600))
			cv2.imshow("Face", mini)
			cv2.imshow("Full", color_frame)

		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			break
		elif k == ord('p'):
			if playing:
				pipeline = recording.pause_playback(pipeline)
			else:
				pipeline = recording.resume_playback(pipeline)
			playing = not playing

	cv2.destroyAllWindows()
	# pipeline.stop()
	pipeline = recording.stop_playback(pipeline)
