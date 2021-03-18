import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
# import colorsys
import time

# import matplotlib.pyplot as plt

# import convert_realsense_to_real_world as coord_conversion
import recording

import human_skeleton
import InfluxDB_connection

import person_identification
import threading

def detect_face_async(human, probabilities_good_face, number_of_faces_detection, color_frame, faces_encodings):
	face_reco = person_identification.recognize_person(human, color_frame, faces_encodings)
	if face_reco is None:
		return
	face_distance, best_match = face_reco
	if skeleton_id not in probabilities_good_face:
		probabilities_good_face[skeleton_id] = 1 - face_distance[best_match]
		number_of_faces_detection[skeleton_id] = 1
	else:
		probabilities_good_face[skeleton_id] += 1 - face_distance[best_match]
		number_of_faces_detection[skeleton_id] += 1


if __name__ == "__main__":

	pipeline = rs.pipeline()
	# pipeline.start()

	align_to = rs.stream.depth
	align = rs.align(align_to)

	backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history = 500)

	images = [
		'faces/marc_blured.jpg',
		'faces/Eric-Benoit.jpg',
		'faces/Alexandre-Benoit.jpg',
		'faces/Stephane-Perrin.jpg',
	]
	labels = [
		'Marc',
		'Eric',
		'Alexandre',
		'Stéphane',
	]
	# faces_encodings = person_identification.create_encodings(images, "faces/all_encodings.encod", verbose = True)
	faces_encodings = person_identification.get_encodings("faces/all_encodings.encod")
	found_persons_names = {}
	probabilities_good_face = {}
	faces_locations = []
	faces_distances = np.array([])
	number_of_faces_detection = {}

	# skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline = pipeline)
	pipeline = recording.start_playback("recordings/walking_normal.bag", pipeline = pipeline, realtime = True)
	skeleton_db2D, skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints("recordings/walking_normal.bag")
	joint_confidence = 0.4

	start_time = time.time()
	num_frames = 0
	human = None
	playing = True

	# automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline)

	while True:
		if playing:
			try:
				# for i in range(331):
				frames = pipeline.wait_for_frames(timeout_ms = 500)
				# playing = False
			except Exception as e:
				break
			num_frames += 1
			# frames = align.process(frames)
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()
			current_timestamp = frames.timestamp

			color_frame = np.asanyarray(color_frame.get_data()).copy()

			if num_frames == 1:
				for i in range(100):
					mask = backSub.apply(color_frame)

			mask = backSub.apply(color_frame)
			original = color_frame.copy()
			# depth_data = np.asanyarray(depth_frame.get_data())

			try:
				for skeleton_id in range(len(skeleton_db2D)):
					human = human_skeleton.HumanSkeleton()
					# human.from_tracker(skeletrack, skeleton_id, depth_frame)
					human.from_json(data2D = skeleton_db2D[skeleton_id], data3D = skeleton_db3D[skeleton_id],
					                specific_timestamp = current_timestamp/1000)

					if human.check_keypoints(['left_shoulder', 'right_shoulder', 'neck']):
						left_shoulder = human.get_keypoint3D('left_shoulder')
						right_shoulder = human.get_keypoint3D('right_shoulder')

						left_to_right_shoulders = right_shoulder - left_shoulder
						neck2D = human.get_keypoint2D('neck')
						neck3D = human.get_keypoint3D('neck')
						looking_right = left_to_right_shoulders.dot(neck3D) > 0

						body_orientation = human.get_body_orientation()
						if body_orientation is not None:
							body_orientation = (body_orientation / np.linalg.norm(body_orientation)) * 400
							body_orientation2D = rs.rs2_project_point_to_pixel(pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(), (neck3D + body_orientation).tolist())

							color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1])), (int(body_orientation2D[0]), int(body_orientation2D[1])), (0, 255, 255), 2)
							if body_orientation[2] > 0:
								color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1] + 5)), (int(body_orientation2D[0]), int(body_orientation2D[1])), (0, 255, 255), 2)
							else:
								color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1] + 5)), (int(body_orientation2D[0]), int(body_orientation2D[1] + 10)), (0, 255, 255), 2)
							legs_overlapping = human.legs_overlapping()

					# recognition_thread = threading.Thread(target = detect_face_async, kwargs = {
					detect_face_async(**{
						'human': human,
						'probabilities_good_face': probabilities_good_face,
						'number_of_faces_detection': number_of_faces_detection,
						'color_frame': color_frame.copy(),
						'faces_encodings': faces_encodings
					})
					# recognition_thread.run()

					if skeleton_id in probabilities_good_face:
						found_persons_names[skeleton_id] = labels[np.argmax(probabilities_good_face[skeleton_id])]
					

					text_position = (-1000, -1000)
					if human.get_center_of_gravity_2d() is not None:
						text_position = tuple(map(int, human.get_center_of_gravity_2d()))



					color_frame = human.display_BBox(color_frame, pipeline, color = (255, 255, 0))
					color_frame[mask > 127] = original[mask > 127]
					color_frame = human.display_BBox(color_frame, pipeline, color = (255, 255, 0), display_back = False)

					if skeleton_id in found_persons_names:
						font = cv2.FONT_HERSHEY_SIMPLEX
						font_size = 0.6
						text_thickness = 2
						text = found_persons_names[skeleton_id] + f"({round((probabilities_good_face[skeleton_id]/number_of_faces_detection[skeleton_id])*100)}%)"
						text_width, text_height = cv2.getTextSize(text, font, font_size, text_thickness)[0]
						
						color_frame = cv2.putText(color_frame,
						                          text,
						                          (int(text_position[0] - text_width/2), int(text_position[1] - text_height/2)),
						                          font, font_size, (255, 255, 0), text_thickness)
			except Exception as e:
				raise e


			cv2.imshow("Camera", cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

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
	pipeline = recording.stop_playback(pipeline)
