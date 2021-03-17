import cv2
import numpy as np
import glob
import skeleton_tracker as st
import pyrealsense2 as rs
import face_recognition # Must be imperatively placed AFTER import pyrealsense2
import numpy as np
import colorsys
import time

import matplotlib.pyplot as plt

# from convert_realsense_to_real_world import *

import human_skeleton
import InfluxDB_connection
import recording

import pickle
import os.path

def create_encodings(images_filenames, save_filename = None, verbose = False):
	all_encodings = []
	if verbose:
		print("Encoding known faces")
	for i, img in enumerate(images):
		if verbose:
			print(f"{round((i / len(images)) * 100)}%")
		try:
			encoding = face_recognition.face_encodings(face_recognition.load_image_file(img))
			all_encodings.append(encoding[0])
		except:
			encoding = face_recognition.face_encodings(face_recognition.load_image_file(img), [(0, 1003, 753, 0)])
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

if __name__ == "__main__":
	images = [
		'faces/marc_blured.jpg',
		'faces/old_man_blured.jpg',
		'faces/old_woman_blured.jpg',
		'faces/midage_man_blured.jpg',
		'faces/midage_woman_blured.jpg',
		'faces/young_man_blured.jpg',
		'faces/young_woman_blured.jpg',
	]
	labels = [
		'Marc',
		'Old Man',
		'Old Woman',
		'Midage Man',
		'Midage Woman',
		'Young Man',
		'Young Woman',
	]

	all_encodings = create_encodings(images, "faces/all_encodings_all_blured.encod", verbose = True)
	# all_encodings = get_encodings("faces/all_encodings.encod")

	align_to = rs.stream.depth
	align = rs.align(align_to)



	pipeline = rs.pipeline()
	# skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline = pipeline)
	pipeline = recording.start_playback("recordings/walking_normal.bag", pipeline = pipeline, realtime = False)
	skeleton_db2D, skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints("recordings/walking_normal.bag")
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
			rgb_small_frame = color_frame.copy() # cv2.resize(color_frame.copy(), (0, 0), fx = 1 / 4, fy = 1 / 4)

			# skeletons = skeletrack.track_skeletons(color_frame) # Live mode

			try:
				# for skeleton_id in range(len(skeletons)):   # Live mode
				for skeleton_id in range(len(skeleton_db2D)): # InfluxDB mode
					human = human_skeleton.HumanSkeleton()
					# human.from_tracker(skeletrack, skeleton_id, depth_frame) # Live mode
					human.from_json(data2D = skeleton_db2D[skeleton_id], data3D = skeleton_db3D[skeleton_id],
					                specific_timestamp = current_timestamp/1000) # InfluxDB mode

					face_locations = [human.get_face_location()]
					face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


					for encoding in face_encodings:
						matches = face_recognition.compare_faces(all_encodings, encoding)

						if len(face_distances) != len(matches):
							face_distances = face_recognition.face_distance(all_encodings, encoding)
						else:
							face_distances += face_recognition.face_distance(all_encodings, encoding)
						best_match_index = np.argmin(face_distances)
						# if matches[best_match_index]:
						found_person_name = labels[best_match_index]
						proba = face_distances[best_match_index]
						print(face_distances)


			except Exception as e:
				print(e)

			color_frame = cv2.putText(color_frame, str(round(proba, 2)) + "%: " + found_person_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

			top, right, bottom, left = 0,1,1,0
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
