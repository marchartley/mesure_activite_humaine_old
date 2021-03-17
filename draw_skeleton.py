import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
import colorsys
import time

import matplotlib.pyplot as plt

from convert_realsense_to_real_world import *

import human_skeleton
import InfluxDB_connection

if __name__ == "__main__":

	pipeline = rs.pipeline()
	# pipeline.start()

	align_to = rs.stream.depth
	align = rs.align(align_to)



	# skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline = pipeline)
	pipeline = start_playback("recordings/walking_normal.bag", pipeline = pipeline, realtime = True)
	skeleton_db2D, skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints("recordings/walking_normal.bag")
	joint_confidence = 0.4

	# backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history=500)

	is_first_step = True
	right_foot_velocity_history = []
	left_foot_velocity_history = []

	right_foot_pos2D = None
	left_foot_pos2D = None
	previous_right_pos = None
	previous_left_pos = None
	right_foot_pos = None
	right_velocity = 0
	left_velocity = 0
	looking_right = False

	last_walking_feet_is_left = True
	number_of_steps = 0

	velocity_threshold = 2.500 # m/sec
	real_velocity_threshold = velocity_threshold # m/frame
	tracing_frames = 1

	num_frame = 0

	start_time = time.time()
	human = None
	playing = True

	current_timestamp = 0
	previous_timestamp = 0
	# automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline)

	while True:
		if playing:
			previous_timestamp = current_timestamp
			try:
				frames = pipeline.wait_for_frames(timeout_ms = 500)
			except Exception as e:
				break
			num_frame += 1
			#frames = align.process(frames)
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()

			current_timestamp = frames.timestamp

			if previous_timestamp > 0:
				real_velocity_threshold = ((current_timestamp - previous_timestamp) * velocity_threshold) / tracing_frames

			color_frame = np.asanyarray(color_frame.get_data()).copy()
			# depth_data = np.asanyarray(depth_frame.get_data())

			# foreground_mask = backSub.apply(color_frame, learningRate = 0.001)
			# color_frame[foreground_mask < 128] = [128, 128, 128]

			# skeletons = skeletrack.track_skeletons_async(color_frame, debug_time = False)
			# for i, human in enumerate(skeletons['body']):
			# 	color_hsv = colorsys.hsv_to_rgb((len(skeletons['ids'])+1) / (i+1), 1.0, 1.0)
			# 	color_rgb = tuple((np.array(color_hsv)*255).tolist())
			# 	for joint in human:
			# 		color_frame = cv2.circle(color_frame, (int(joint[0]), int(joint[1])), 4, color_rgb, 4)


			try:
				for skeleton_id in range(len(skeleton_db2D)):
					human = human_skeleton.HumanSkeleton()
					# human.from_tracker(skeletrack, skeleton_id, depth_frame)
					human.from_json(data2D = skeleton_db2D[skeleton_id], data3D = skeleton_db3D[skeleton_id], specific_timestamp = current_timestamp/1000)
					right_foot_pos2D = human.get_keypoint2D('right_ankle')
					right_foot_pos = human.get_keypoint3D('right_ankle')

					left_foot_pos2D = human.get_keypoint2D('left_ankle')
					left_foot_pos = human.get_keypoint3D('left_ankle')

					left_shoulder = human.get_keypoint3D('left_shoulder')
					right_shoulder = human.get_keypoint3D('right_shoulder')

					left_to_right_shoulders = right_shoulder - left_shoulder
					neck2D = human.get_keypoint2D('neck')
					neck3D = human.get_keypoint3D('neck')
					looking_right = left_to_right_shoulders.dot(neck3D) > 0

					body_orientation = human.get_body_orientation()
					body_orientation = (body_orientation / np.linalg.norm(body_orientation)) * 400
					body_orientation2D = rs.rs2_project_point_to_pixel(pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(), (neck3D + body_orientation).tolist())

					color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1])), (int(body_orientation2D[0]), int(body_orientation2D[1])), (0, 255, 255), 2)
					if body_orientation[2] > 0:
						color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1] + 5)), (int(body_orientation2D[0]), int(body_orientation2D[1])), (0, 255, 255), 2)
					else:
						color_frame = cv2.line(color_frame, (int(neck2D[0]), int(neck2D[1] + 5)), (int(body_orientation2D[0]), int(body_orientation2D[1] + 10)), (0, 255, 255), 2)
					legs_overlapping = human.legs_overlapping()

					# 	print("------------------ RIGHT")
					# else:
					# 	print("LEFT -------------------")
					# if human.legs_overlapping():
					# 	print("---------- OVERLAPPPP ------------")
					# else:
					# 	print("----------            ------------")

					if num_frame % tracing_frames == 0:
						if previous_right_pos is not None:
							right_velocity = np.array(right_foot_pos) - np.array(previous_right_pos)
							right_velocity = int(np.linalg.norm(right_velocity) / tracing_frames)
						previous_right_pos = right_foot_pos

						if previous_left_pos is not None:
							left_velocity = np.array(left_foot_pos) - np.array(previous_left_pos)
							left_velocity = int(np.linalg.norm(left_velocity) / tracing_frames)
						previous_left_pos = left_foot_pos


						if looking_right and legs_overlapping:
							right_velocity = 0
						if right_velocity > 2000:
							if len(right_foot_velocity_history) > 0:
								right_velocity = right_foot_velocity_history[-1]
							else:
								right_velocity = 0

						if not looking_right and legs_overlapping:
							left_velocity = 0
						if left_velocity > 2000:
							if len(left_foot_velocity_history) > 0:
								left_velocity = left_foot_velocity_history[-1]
							else:
								left_velocity = 0

						right_foot_velocity_history.append(right_velocity)
						left_foot_velocity_history.append(left_velocity)

						# if  skeletons['confidence'][0][10] < joint_confidence: # Right ankle, need to be changed to be understandable
						# 	right_velocity = 0
						# if  skeletons['confidence'][0][13] < joint_confidence: # Left ankle, need to be changed to be understandable
						# 	left_velocity = 0

						if left_velocity > right_velocity:
							right_velocity = 0
						elif left_velocity < right_velocity:
							left_velocity = 0

						if left_velocity >= real_velocity_threshold and (not last_walking_feet_is_left or is_first_step):
							is_first_step = False
							number_of_steps += 1
							last_walking_feet_is_left = True
						elif right_velocity >= real_velocity_threshold and (last_walking_feet_is_left or is_first_step):
							is_first_step = False
							number_of_steps += 1
							last_walking_feet_is_left = False


			except Exception as e:
				pass
				# print(e)
				# if str(e) == "get_keypoint3D() missing 1 required positional argument: 'depth_map'":
				# 	raise (e)

			# if human is not None:
			# 	color = (255, 0, 0) if human.legs_overlapping() else (0, 0, 255)
			# 	for key in human.body_parts_2D:
			# 		color_frame = cv2.circle(color_frame, (int(human.body_parts_2D[key][0]), int(human.body_parts_2D[key][1])), 4, color, 4)
			# try:
			# 	print(number_of_steps)
			# except:
			# 	pass
			# print(number_of_steps)
			try:
				if right_foot_pos2D is not None:
					color = (0, 0, 255)
					if right_velocity >= real_velocity_threshold:
						if human.legs_overlapping():#if looking_right and human.legs_overlapping():
							color = (0, 255, 0)
						else:
							color = (255, 0, 0)
					color_frame = cv2.circle(color_frame, (int(right_foot_pos2D[0]), int(right_foot_pos2D[1])), 4, color, 4)
				if left_foot_pos2D is not None:
					color = (0, 0, 255)
					if left_velocity >= real_velocity_threshold:
						if human.legs_overlapping():#if not looking_right and human.legs_overlapping():
							color = (0, 255, 0)
						else:
							color = (255, 0, 0)
					color_frame = cv2.circle(color_frame, (int(left_foot_pos2D[0]), int(left_foot_pos2D[1])), 4, color, 4)
			# if right_foot_pos2D is not None:
			# 	color = (0, 0, 255) if not human.legs_overlapping() else (0, 255, 0) #if last_walking_feet_is_left else (255, 0, 0)
			# 	color_frame = cv2.circle(color_frame, (int(right_foot_pos2D[0]), int(right_foot_pos2D[1])), 4, color, 4)
			# if left_foot_pos2D is not None:
			# 	color = (0, 0, 255) if not human.legs_overlapping() else (0, 255, 0) #if not last_walking_feet_is_left else (255, 0, 0)
			# 	color_frame = cv2.circle(color_frame, (int(left_foot_pos2D[0]), int(left_foot_pos2D[1])), 4, color, 4)
				color_frame = cv2.putText(color_frame, str(number_of_steps), (int(left_foot_pos2D[0] + 4), int(left_foot_pos2D[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
			except:
				pass


			# try:
			# 	thickness1 = 20
			# 	thickness2 = 20
			# 	start1 = human.get_keypoint2D('left_hip')
			# 	end1 = human.get_keypoint2D('left_knee')
			# 	start2 = human.get_keypoint2D('right_hip')
			# 	end2 = human.get_keypoint2D('right_knee')
			# 	start1 = np.array(start1[:2])
			# 	end1 = np.array(end1[:2])
			# 	start2 = np.array(start2[:2])
			# 	end2 = np.array(end2[:2])
            #
			# 	AB1 = end1 - start1
			# 	AB2 = end2 - start2
            #
			# 	normalized_AB1 = AB1 / np.linalg.norm(AB1)
			# 	normalized_AB2 = AB2 / np.linalg.norm(AB2)
            #
			# 	rotated_AB1 = np.array([normalized_AB1[1], normalized_AB1[0]])
			# 	rotated_AB2 = np.array([normalized_AB2[1], normalized_AB2[0]])
            #
			# 	left_part1 = np.array([start1 - normalized_AB1 * thickness1 + rotated_AB1 * thickness1/2, end1 + normalized_AB1 * thickness1 + rotated_AB1 * thickness1/2])
			# 	right_part1 = np.array([start1 - normalized_AB1 * thickness1 - rotated_AB1 * thickness1/2, end1 + normalized_AB1 * thickness1 - rotated_AB1 * thickness1/2])
			# 	left_part2 = np.array([start2 - normalized_AB2 * thickness2 + rotated_AB2 * thickness2/2, end2 + normalized_AB2 * thickness2 + rotated_AB2 * thickness2/2])
			# 	right_part2 = np.array([start2 - normalized_AB2 * thickness2 - rotated_AB2 * thickness2/2, end2 + normalized_AB2 * thickness2 - rotated_AB2 * thickness2/2])
            #
			# 	color_frame = cv2.line(color_frame, tuple(map(int, left_part1[0])), tuple(map(int, left_part1[1])), (255, 255, 0), 3)
			# 	color_frame = cv2.line(color_frame, tuple(map(int, right_part1[0])), tuple(map(int, right_part1[1])), (255, 0, 255), 3)
			# except Exception as e:
			# 	print(e)

			cv2.imshow("Camera", cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			break
		elif k == ord('+'):
			velocity_threshold += 5
		elif k == ord('-'):
			velocity_threshold -= 5
		elif k == ord('p'):
			if playing:
				pipeline = pause_playback(pipeline)
			else:
				pipeline = resume_playback(pipeline)
			playing = not playing


		# print(f"Velocity : {str(velocity_threshold).rjust(5, ' ')}; Frame rate: {round(((start_time - time.time())/num_frame)*1000)}ms")
	cv2.destroyAllWindows()
	pipeline = stop_playback(pipeline)
	# pipeline.stop()

	plt.plot(right_foot_velocity_history, 'r-')
	plt.plot(left_foot_velocity_history, 'b-')
	plt.show()
