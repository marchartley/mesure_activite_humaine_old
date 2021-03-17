import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
import colorsys
import time
import json

import matplotlib.pyplot as plt

from convert_realsense_to_real_world import *

import human_skeleton

def save_data(bag_filename, saving_filename, visual=False):
	pipeline = rs.pipeline()

	skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline = pipeline)
	pipeline = start_playback(bag_filename, pipeline = pipeline, realtime = False)
	joint_confidence = 0.4


	coordinates = {}
	distances = {}
	velocities = {}
	timestamps = []
	frame_indexes = []

	velocity_threshold = 2.500 # m/sec
	real_velocity_threshold = velocity_threshold # m/frame
	tracing_frames = 3

	num_frame = 0

	start_time = time.time()
	human = None

	current_timestamp = 0
	previous_timestamp = 0

	print("Starting...")

	while True:
		try:
			previous_timestamp = current_timestamp
			frames = pipeline.wait_for_frames(timeout_ms = 100)
			num_frame += 1
			color_frame = frames.get_color_frame()
			depth_frame = frames.get_depth_frame()

			current_timestamp = frames.timestamp

			if previous_timestamp > 0:
				real_velocity_threshold = (current_timestamp - previous_timestamp) * velocity_threshold


			# if not color_frame or not depth_frame: continue

			color_frame = np.asanyarray(color_frame.get_data()).copy()

			skeletons = skeletrack.track_skeletons(color_frame, debug_time = False)

			if skeletrack.check_user_presence(color_frame):
				human = human_skeleton.HumanSkeleton()
				human.from_tracker(skeletrack, 0, depth_frame)

				coord2D, coord3D = human.to_json()

				for key in coord3D:
					if key not in coordinates:
						coordinates[key] = []
						velocities[key] = []

						for key2 in coord3D:
							distances[key + "-" + key2] = []

					if np.any(coord2D[key] == -1):
						coordinates[key].append(np.array([None, None, None]))
						velocities[key].append(None)
						for key2 in coord3D:
							distances[key + "-" + key2].append(None)


						continue #coord3D[key] = np.array([None, None, None])

					coordinates[key].append(coord3D[key])
					if len(coordinates[key]) == 1:
						velocities[key].append(0)
					else:
						if np.any(coordinates[key][-2] == None) or np.any(coordinates[key][-1] == None):
							velocities[key].append(None)
						else:
							velocities[key].append(np.linalg.norm((coordinates[key][-2] - coordinates[key][-1]) / (current_timestamp - previous_timestamp)))


					for key2 in coord3D:
						distances[key + "-" + key2].append(np.linalg.norm(coord3D[key2] - coord3D[key]))


				timestamps.append(current_timestamp)
				frame_indexes.append(num_frame)

		except Exception as e:
			break

	# velocities = {}
	# for key in coordinates:
	# 	for i in range(len(coordinates[key])):
	# 		if coordinates[key][i] !=

	non_ndarray_coord = {}
	for key in coordinates:
		non_ndarray_coord[key] = []
		for coord in coordinates[key]:
			non_ndarray_coord[key].append(coord.tolist())

	data = {
	# 'right' : right_foot_velocity_history,
	# 'left' : left_foot_velocity_history,
	# 'distance' : feet_distance_history,
	'coordinates': non_ndarray_coord,
	'velocities' : velocities,
	'distances'  : distances,
	'timestamps' : timestamps,
	'frames' : frame_indexes
	}
	# print(data)
	with open(saving_filename, 'w+') as file:
		json.dump(data, file)

	print("Done.")


	pipeline = stop_playback(pipeline)
	# pipeline.stop()

def estimate_footsteps(data, motion_threshold = 10):
	left_data = data['left']
	right_data = data['right']
	timestamps = data['timestamps']
	frames = data['frames']
	estimated_timestamps = []
	estimated_frames = []

	number_of_steps = 0
	current_foot_walking = None
	for (left, right, timestamp, frame) in zip(left_data, right_data, timestamps, frames):
		if left > motion_threshold and current_foot_walking != "left":
			current_foot_walking = "left"
			number_of_steps += 1
			estimated_timestamps.append(timestamp)
			estimated_frames.append(frame)
		elif right > motion_threshold and current_foot_walking != "right":
			current_foot_walking = "right"
			number_of_steps += 1
			estimated_timestamps.append(timestamp)
			estimated_frames.append(frame)
	return number_of_steps, {'timestamps': np.array(estimated_timestamps), 'frames': np.array(estimated_frames)}

def estimate_footsteps_by_distance(data, distance_range = 2):
	distance = data['distance']
	timestamps = data['timestamps']
	frames = data['frames']
	estimated_timestamps = []
	estimated_frames = []
	number_of_steps = 0

	for i in range(1, len(distance)-1):
		# If dist[i] is a peak, count as a step
		#print(i, ':', np.all(distance[i - distance_range: i] < distance[i]), ' ', np.all(distance[i +1 : i + distance_range + 1] < distance[i]))
		if distance[i] > 0 and not np.any(distance[max(0, i - distance_range): min(len(distance), i + distance_range + 1)] > distance[i]): #distance[i - 1] < distance[i] and distance[i + 1] < distance[i]:
			number_of_steps += 1
			estimated_timestamps.append(timestamps[i])
			estimated_frames.append(frames[i])

	return number_of_steps, {'timestamps': np.array(estimated_timestamps), 'frames': np.array(estimated_frames)}


def get_data(json_filename):
	with open(json_filename, 'r') as file:
		return json.load(file)

def truncate_values(values, limit, convert_None_to=None):
	return np.array([convert_None_to if val is None else val if val < limit else limit for val in values])

def replace_limit_values_with_previous(values, limit, convert_None_to=None):
	new_vals = []
	values = truncate_values(values, limit)
	for i in range(len(values)):
		if i == 0:
			new_vals.append(values[i])
		else:
			new_vals.append(convert_None_to if values[i] is None else values[i] if values[i] < limit else new_vals[-1])
	return np.array(new_vals)

def mean_smooth_data(values, delta):
	smoothed = []
	for i in range(len(values)):
		low_bound = max(0, i-delta)
		high_bound = min(len(values)-1, i + delta + 1)
		try:
			smoothed.append(np.mean(values[low_bound:high_bound]))
		except:
			smoothed.append(0)
	return np.array(smoothed)

def remove_None_by_interpolation(values, extrem_values = 0):
	for i in range(len(values)):
		if values[i] is None:
			distance_before = 0
			distance_after = 0
			value_before = extrem_values
			value_after = extrem_values
			for j in range(i, 0, -1):
				if values[j] is not None:
					value_before = values[j]
					break
				else:
					distance_before += 1
			for j in range(i, len(values)):
				if values[j] is not None:
					value_after = values[j]
					break
				else:
					distance_after += 1

			values[i] = value_before * (distance_before / (distance_before + distance_after)) + value_after * (distance_after / (distance_before + distance_after))
	return values


def visualize_data(data, time_in_seconds = True):
	footsteps_motion = estimate_footsteps(data, motion_threshold = 0.2)[1]
	footsteps_distance = estimate_footsteps_by_distance(data, 1)[1]

	timestamps = data['timestamps']
	frames = data['frames']
	timestamps = ((timestamps - timestamps[0]) / 1000)

	if time_in_seconds:
		x_axis = timestamps
		x_label = "Time (s)"
		motion_detection_motion = footsteps_motion['timestamps']
		motion_detection_distance = footsteps_distance['timestamps']
	else:
		x_axis = frames
		x_label = "Time (frames)"
		motion_detection_motion = footsteps_motion['frames']
		motion_detection_distance = footsteps_distance['frames']


	plt.subplot(211)
	plt.plot(x_axis, data['left'], 'r-', label='Left foot')
	plt.plot(x_axis, data['right'], 'b-', label='Right foot')
	for foot_estimation in motion_detection_motion:
		plt.axvline(x=foot_estimation, color = (.0, .0, 1.0, .4))
	plt.title("Difference of velocity per foot in time")
	plt.ylabel("Speed (m/s)")
	plt.xlabel(x_label)
	plt.legend()


	plt.subplot(212)
	plt.plot(x_axis, data['distance']/10, 'g-', label='Distance between feet')
	for foot_estimation in motion_detection_distance:
		plt.axvline(x=foot_estimation, color = (.0, .0, 1.0, .4))
	plt.ylabel('Distance (cm)')
	plt.title("Distance between feet in time")
	plt.xlabel(x_label)
	plt.legend()


	plt.show()

if __name__ == "__main__":
	# save_data("recordings/walking_slow.bag", "data_2021_03_03_slow.json", visual=True)
	data = get_data("data_2021_03_03_normal.json")

	motion_threshold = 0.1
	# save_data("A:/Code Perso/NVIDIA_realsense/test_project/video_samples/walk_around_aruco.bag", "data_2021_03_01_walk_around_aruco.json", visual=True)
	# data = get_data("data_2021_03_01_walk_around_aruco.json")


	left = remove_None_by_interpolation(np.array(data['velocities']['LEFT_ANKLE']))
	right = remove_None_by_interpolation(np.array(data['velocities']['RIGHT_ANKLE']))
	distance =  remove_None_by_interpolation(np.array(data['distances']['RIGHT_ANKLE-LEFT_ANKLE']))

	right_hip = remove_None_by_interpolation(np.array(data['velocities']['RIGHT_HIP']))
	left_hip = remove_None_by_interpolation(np.array(data['velocities']['LEFT_HIP']))

	# left = truncate_values(left, 20)
	# right = truncate_values(right, 20)
	# distance = truncate_values(distance, 500)
	# left = replace_limit_values_with_previous(left, 20)
	# right = replace_limit_values_with_previous(right, 20)
	# distance = replace_limit_values_with_previous(distance, 2000)
	left = mean_smooth_data(left, 2)
	right = mean_smooth_data(right, 2)
	distance = mean_smooth_data(distance, 2)
	left = mean_smooth_data(left, 2)
	right = mean_smooth_data(right, 2)
	distance = mean_smooth_data(distance, 2)

	timestamps = np.array(data['timestamps']) - data['timestamps'][0] # Begin timestamps at 0
	frames = np.array(data['frames'])

	diff_right = right - left
	diff_left = left - right


	diff_right[diff_right < 0] = 0
	diff_left[diff_left < 0] = 0

	smoothed_data = {
		'right' : diff_right,
		'left' : diff_left,
		'distance' : distance,
		'timestamps' : timestamps,
		'frames' : frames
	}

	footsteps_motion = estimate_footsteps(smoothed_data, motion_threshold = motion_threshold)[0]
	footsteps_distance = estimate_footsteps_by_distance(smoothed_data, 1)[0]
	print(footsteps_motion, "footsteps estimated with motion")
	print(footsteps_distance, "footsteps estimated with distance")
	# for i in range(20):
	# 	print(estimate_footsteps_by_distance(smoothed_data, i)[0], "footsteps estimated with distance [i = ", i, "]")

	print("Cycle Time                 = ", footsteps_motion / ((timestamps[-1] - timestamps[0])/1000))
	print("Stance/Swing ratio (Right) = ", (len(diff_right) / len(diff_right[diff_right <= motion_threshold])))
	print("Stance/Swing ratio (Left)  = ", (len(diff_left) / len(diff_left[diff_left <= motion_threshold])))
	print("Double support time        = ", len(diff_right[abs(diff_right - diff_left) < motion_threshold] < motion_threshold) / len(diff_right))

	dist_hip_right = (right_hip.mean() * (timestamps[-1] - timestamps[0]))/1000
	dist_hip_left = (left_hip.mean() * (timestamps[-1] - timestamps[0]))/1000
	print("Distance walked            = ", (dist_hip_left + dist_hip_right)/2, "m")
	print("Distance by footstep       = ", ((dist_hip_left + dist_hip_right)/2) / footsteps_motion, "m/footstep")
	# visualize_data(smoothed_data, time_in_seconds=False)

