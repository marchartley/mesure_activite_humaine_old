import cv2
import numpy as np
import pyrealsense2 as rs

from recording import *

from cv2 import aruco

import realsense_filters as filters

###############################################
# marker generation

aruco_box = {
	0: {
		'vertical': 'y',
		'horizontal': 'x',
		'inverse_third_dim': True
	},
	1: {
		'vertical': 'y',
		'horizontal': '-z',
		'inverse_third_dim': False
	},
	2: {
		'vertical': 'y',
		'horizontal': '-x',
		'inverse_third_dim': True
	},
	3: {
		'vertical': 'y',
		'horizontal': 'z',
		'inverse_third_dim': False
	},
	4: {
		'vertical': 'z',
		'horizontal': 'x',
		'inverse_third_dim': False
	},
	5: {
		'vertical': 'z',
		'horizontal': '-x',
		'inverse_third_dim': False
	}
}


def generateMarkers(aruco_dict, n_x = 1, n_y = 1):
	border = 100
	size = 300

	tags = []

	aruco_target = np.zeros((border * (n_y + 1) + size * n_y, border * (n_x + 1) + size * n_x))
	aruco_target[::] = 255

	for i in range(n_y * n_x):
		img = aruco.drawMarker(aruco_dict, i, size)
		tags.append(img)
		y_offset = border + (i // n_x) * (size + border)
		x_offset = border + (i % n_x) * (size + border)
		aruco_target[y_offset:y_offset + img.shape[0],
		x_offset:x_offset + img.shape[1]] = img

	return tags


###############################################
# marker detection


def detectMarker(frame, aruco_dict):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	parameters = aruco.DetectorParameters_create()
	corners, ids, rejectedImgPoints = aruco.detectMarkers(
		gray, aruco_dict, parameters = parameters)

	# SUB PIXEL DETECTION
	criteria = (cv2.TERM_CRITERIA_EPS +
	            cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
	for corner in corners:
		cv2.cornerSubPix(gray, corner, winSize = (
			3, 3), zeroZone = (-1, -1), criteria = criteria)

	frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
	return frame_markers, corners, ids


def get_3D_coord_from_realsense(pixel_coord, depth_frame, pipeline):
	depth_stream = pipeline.get_active_profile().get_stream(rs.stream.depth)
	color_stream = pipeline.get_active_profile().get_stream(rs.stream.color)

	depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_option(rs.option.depth_units)  # depth_sensor.get_depth_scale()

	depth_stream_profile = depth_stream.as_video_stream_profile()
	color_stream_profile = color_stream.as_video_stream_profile()

	min_depth = 0.01
	max_depth = 16.0

	depth_intrin = depth_stream_profile.get_intrinsics()
	color_intrin = color_stream_profile.get_intrinsics()

	depth_to_color_extrin = depth_stream_profile.get_extrinsics_to(color_stream_profile)
	color_to_depth_extrin = color_stream_profile.get_extrinsics_to(depth_stream_profile)

	pixel_coord = np.array(pixel_coord, int).tolist()

	color_point = rs.rs2_deproject_pixel_to_point(color_intrin, pixel_coord, 1)
	depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, color_point)
	depth_coord = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)

	# depth_coord = rs.rs2_project_color_pixel_to_depth_pixel(
	#     depth_frame.get_data(), depth_scale,
	#     min_depth, max_depth,
	#     depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, pixel_coord)
	distance = depth_frame.get_distance(int(depth_coord[0]), int(depth_coord[1]))

	return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, depth_coord, distance))  # / depth_scale


def back_to_pixel(point_coord, pipeline, old_origin_coord = None, old_x_axis = None, old_y_axis = None,
                  old_z_axis = None):
	if old_origin_coord is None:
		old_origin_coord = [0, 0, 0]
	if old_x_axis is None:
		old_x_axis = [1, 0, 0]
	if old_y_axis is None:
		old_y_axis = [0, 1, 0]
	color_intrin = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

	x_axis = np.array(old_x_axis) - np.array(old_origin_coord)
	y_axis = np.array(old_y_axis) - np.array(old_origin_coord)
	if old_z_axis is None:
		z_axis = compute_z_axis(x_axis, y_axis)
	else:
		z_axis = np.array(old_z_axis) - np.array(old_origin_coord)

	transform = np.matrix([x_axis, y_axis, z_axis])
	coord_from_cam = np.dot(point_coord, transform)
	coord_from_cam += old_origin_coord

	return rs.rs2_project_point_to_pixel(color_intrin, coord_from_cam.tolist()[0])


def translate_coord_to_world(coord, new_origin_coord, new_x_axis_coord, new_y_axis_coord, new_z_axis_coord = None):
	original_X = np.matrix([1, 0, 0]).transpose()
	original_Y = np.matrix([0, 1, 0]).transpose()
	original_Z = np.matrix([0, 0, 1]).transpose()
	M1 = np.column_stack((original_X, original_Y, original_Z))  # It's just the identity matrix
	p1 = np.matrix(coord).transpose()
	O1 = np.matrix([0, 0, 0]).transpose()

	O2 = np.matrix(new_origin_coord).transpose()
	new_X = np.matrix(new_x_axis_coord).transpose() - O2
	new_Y = np.matrix(new_y_axis_coord).transpose() - O2
	if new_z_axis_coord is not None:
		new_Z = np.matrix(new_z_axis_coord).transpose() - O2
	else:
		new_Z = compute_z_axis(new_X, new_Y)
	M2 = np.column_stack((new_X, new_Y, new_Z))  # It's just the identity matrix

	p2 = np.dot(np.linalg.inv(M2), ((O1 - O2) + np.dot(M1, p1)))  # p2 = M2' * (O1 - O2 + M1 * p1)
	return p2


def compute_z_axis(x_axis, y_axis):
	z_axis = np.cross(x_axis.transpose(), y_axis.transpose()).transpose()
	# Getting a coherent Z norm
	norm_X = np.linalg.norm(x_axis)
	norm_Y = np.linalg.norm(y_axis)
	norm_Z = np.linalg.norm(z_axis)
	z_axis = (z_axis / norm_Z) * ((norm_X + norm_Y) / 2)
	return z_axis


def draw_line(image, start, end, color = (0, 0, 0), thickness = 2):
	image = cv2.line(image, tuple(map(int, start)), tuple(map(int, end)), color = color, thickness = thickness)
	return image


all_top_left = []


def draw_cube(image, marker, depth_frame, pipeline):
	(top_left, top_right, bottom_right, bottom_left) = (marker[0][0], marker[0][1], marker[0][2], marker[0][3])

	top_left_3D = get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
	top_right_3D = get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
	bottom_left_3D = get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

	if np.all(top_left_3D == 0) or np.all(top_right_3D == 0) or np.all(bottom_left_3D == 0):
		return image

	# color_intrin = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

	p000 = back_to_pixel([0, 0, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p001 = back_to_pixel([0, 0, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p010 = back_to_pixel([0, 1, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p011 = back_to_pixel([0, 1, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p100 = back_to_pixel([1, 0, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p101 = back_to_pixel([1, 0, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p110 = back_to_pixel([1, 1, 0], pipeline, top_left_3D, top_right_3D, bottom_left_3D)
	p111 = back_to_pixel([1, 1, 1], pipeline, top_left_3D, top_right_3D, bottom_left_3D)

	# Depth
	image = draw_line(image, p000, p001, (0, 255, 0))
	image = draw_line(image, p010, p011, (0, 255, 0))
	image = draw_line(image, p100, p101, (0, 255, 0))
	image = draw_line(image, p110, p111, (0, 255, 0))

	# Vertical
	image = draw_line(image, p000, p010, (255, 0, 0))
	image = draw_line(image, p001, p011, (255, 0, 0))
	image = draw_line(image, p100, p110, (255, 0, 0))
	image = draw_line(image, p101, p111, (255, 0, 0))

	# Horizontal
	image = draw_line(image, p000, p100, (0, 0, 255))
	image = draw_line(image, p001, p101, (0, 0, 255))
	image = draw_line(image, p010, p110, (0, 0, 255))
	image = draw_line(image, p011, p111, (0, 0, 255))

	image = draw_line(image, top_left, top_right, (0, 0, 255), thickness = 1)
	image = draw_line(image, bottom_right, bottom_left, (0, 0, 255), thickness = 1)
	image = draw_line(image, top_left, bottom_left, (255, 0, 0), thickness = 1)
	image = draw_line(image, top_right, bottom_right, (255, 0, 0), thickness = 1)
	return image

def center_of_shape(vertices:list):
	return np.mean(np.array(vertices), axis=0)

def create_aruco_cube(corners, ids, depth_frame, pipeline):
	vectors_counts = {'x': 0, 'y': 0, 'z': 0}
	reference_vectors = {'x': np.array([0.0, 0.0, 0.0]),
	                     'y': np.array([0.0, 0.0, 0.0]),
	                     'z': np.array([0.0, 0.0, 0.0])}
	reference_center = np.array([0, 0, 0])
	if len(corners) == 0:
		return

	ids = np.array(ids).flatten()
	centers3D = []
	for i, corner in enumerate(corners):
		(top_left, top_right, bottom_right, bottom_left) = (corner[0][0], corner[0][1], corner[0][2], corner[0][3])
		center2D = center_of_shape(corner[0])
		centers3D.append(get_3D_coord_from_realsense(center2D, depth_frame, pipeline))

		top_left_3D = get_3D_coord_from_realsense(top_left, depth_frame, pipeline)
		top_right_3D = get_3D_coord_from_realsense(top_right, depth_frame, pipeline)
		bottom_left_3D = get_3D_coord_from_realsense(bottom_left, depth_frame, pipeline)
		bottom_right_3D = get_3D_coord_from_realsense(bottom_right, depth_frame, pipeline)

		# Get axis associated with this marker
		v_ax = aruco_box[ids[i]]['vertical'][-1] # last char is "x", "y" or "z"
		v_sign = -1 if aruco_box[ids[i]]['vertical'][0] == "-" else 1
		h_ax = aruco_box[ids[i]]['horizontal'][-1] # last char is "x", "y" or "z"
		h_sign = -1 if aruco_box[ids[i]]['horizontal'][0] == "-" else 1

		last_ax = 'x' if 'x' not in (v_ax, h_ax) else 'y' if 'y' not in (v_ax, h_ax) else 'z'
		last_sign = -1 if aruco_box[ids[i]]['inverse_third_dim'] else 1

		horizontal_axis = np.mean(np.array([top_left_3D - top_right_3D, bottom_left_3D - bottom_right_3D]), axis = 0) * h_sign
		vertical_axis = np.mean(np.array([top_left_3D - bottom_left_3D, top_right_3D - bottom_right_3D]), axis = 0) * v_sign

		reference_vectors[v_ax] += vertical_axis
		reference_vectors[h_ax] += horizontal_axis
		reference_vectors[last_ax] += compute_z_axis(vertical_axis, horizontal_axis) * last_sign

		vectors_counts[v_ax] += 1
		vectors_counts[h_ax] += 1
		vectors_counts[last_ax] += 1

	for axis in reference_vectors:
		if vectors_counts[axis] > 0:
			reference_vectors[axis] /= vectors_counts[axis] # Mean of each axis

	for i in range(len(centers3D)):
		ax_v = aruco_box[ids[i]]['vertical'][-1]
		ax_h = aruco_box[ids[i]]['horizontal'][-1]

		v_dir = 1 if aruco_box[ids[i]]['vertical'][0] == "-" else 0
		h_dir = 1 if aruco_box[ids[i]]['horizontal'][0] == "-" else 0
		inv3dim = 1 if aruco_box[ids[i]]['inverse_third_dim'] else 0

		# Guess I have to understand why is that, but for now it works
		# If there is exactly 1 "negative" axis, you have to invert the offset
		center_offset = -1 if (v_dir + h_dir + inv3dim) % 2 == 0 else 1

		if 'x' not in (ax_v, ax_h):
			centers3D[i] += reference_vectors['x'] * 0.5 * center_offset
		if 'y' not in (ax_v, ax_h):
			centers3D[i] += reference_vectors['y'] * 0.5 * center_offset
		if 'z' not in (ax_v, ax_h):
			centers3D[i] += reference_vectors['z'] * 0.5 * center_offset

	mean_center = np.mean(np.array(centers3D), axis = 0)

	print(f"Calculated with {len(centers3D)} aruco codes")

	return mean_center, reference_vectors


pure_reference = {
	'x_vector': None,
	'y_vector': None,
	'z_vector': None
}
if __name__ == "__main__":
	# Example of usage: python convert_realsense_to_real_world.py --file recordings/aruco_code.bag
	import argparse

	parser = argparse.ArgumentParser(description = 'Transform the coordinate reference to fit an Aruco marker')
	parser.add_argument('-f', '--filename', dest = 'filename', metavar = 'filename', type = str,
	                    help = 'Play a specific sequence .bag', required = False)
	args = parser.parse_args()
	filename = args.filename

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	pipeline = rs.pipeline()
	if filename is None:
		pipeline.start()
	else:
		pipeline = start_playback(filename, pipeline, realtime = True, repeat_playback = True)

	aruco_origin = [0, 0, 0]

	align_to = rs.stream.depth
	align = rs.align(align_to)

	filters.create_custom_filters("custom1", [
		rs.temporal_filter(smooth_alpha = 0.060, smooth_delta = 50, persistence_control = 8)])

	marker_size = 0.08
	while True:
		try:
			frames = pipeline.wait_for_frames()
		except:
			break

		# frames = align.process(frames)
		color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
		depth_frame = filters.aruco_detection_filters(frames.get_depth_frame()).as_depth_frame()
		# depth_frame = filters.custom_filters(frames.get_depth_frame(), "custom1").as_depth_frame()
		# depth_frame = frames.get_depth_frame()

		marked_frame, corners, ids = detectMarker(color_frame, aruco_dict)

		depth_image = np.array(rs.colorizer(2).colorize(depth_frame).get_data())
		if len(corners) > 0:
			try:
				center, vectors = create_aruco_cube(corners, ids, depth_frame, pipeline)
				center2D = tuple(map(int, back_to_pixel(center, pipeline)))
				color_frame = cv2.circle(color_frame, center2D, 3, (0, 255, 0), -1)

				x_axis = tuple(map(int, back_to_pixel(center + vectors['x'] * 2, pipeline)))
				y_axis = tuple(map(int, back_to_pixel(center + vectors['y'] * 2, pipeline)))
				z_axis = tuple(map(int, back_to_pixel(center + vectors['z'] * 2, pipeline)))
				color_frame = cv2.line(color_frame, center2D, x_axis, (255, 0, 0), 4)
				color_frame = cv2.line(color_frame, center2D, y_axis, (0, 0, 255), 4)
				color_frame = cv2.line(color_frame, center2D, z_axis, (0, 255, 0), 4)
			except Exception as e:
				print(e)
			# ids = ids.flatten()
			# # color_frame = draw_cube(color_frame, corners[0], depth_frame, pipeline)
			# for corner, aruco_id in zip(corners, ids):
			# 	center = center_of_shape(corner[0])
			# 	center_3D = get_3D_coord_from_realsense(center, depth_frame, pipeline)
			# 	try:
			# 		color_frame = draw_cube(color_frame, corner, depth_frame, pipeline)
			# 	except Exception as e:
			# 		pass

		color_frame = cv2.resize(cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB), (800, 600))
		cv2.imshow("color", color_frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if filename is None:
		pipeline = pipeline.stop()
	else:
		pipeline = stop_playback(pipeline)
