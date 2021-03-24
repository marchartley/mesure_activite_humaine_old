import cv2
import numpy as np
import pyrealsense2 as rs

from recording import *

from cv2 import aruco

import realsense_filters as filters

###############################################
# marker generation


def generateMarkers(aruco_dict, n_x = 1, n_y = 1):
    border = 100
    size = 300

    tags = []

    aruco_target = np.zeros((border*(n_y+1)+size*n_y, border*(n_x+1)+size*n_x))
    aruco_target[::] = 255

    for i in range(n_y*n_x):
        img = aruco.drawMarker(aruco_dict, i, size)
        tags.append(img)
        y_offset = border+(i//n_x)*(size+border)
        x_offset = border+(i % n_x)*(size+border)
        aruco_target[y_offset:y_offset+img.shape[0],
                     x_offset:x_offset+img.shape[1]] = img

    return tags

###############################################
# marker detection


def detectMarker(frame, aruco_dict):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    # SUB PIXEL DETECTION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize=(
            3, 3), zeroZone=(-1, -1), criteria=criteria)

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    return frame_markers, corners, ids


def get_3D_coord_from_realsense(pixel_coord, depth_frame, pipeline):
	depth_stream = pipeline.get_active_profile().get_stream(rs.stream.depth)
	color_stream = pipeline.get_active_profile().get_stream(rs.stream.color)

	depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()

	depth_stream_profile = depth_stream.as_video_stream_profile()
	color_stream_profile = color_stream.as_video_stream_profile()

	min_depth = 0.01
	max_depth = 16.0

	depth_intrin = depth_stream_profile.get_intrinsics()
	color_intrin = color_stream_profile.get_intrinsics()

	depth_to_color_extrin = depth_stream_profile.get_extrinsics_to(color_stream)
	color_to_depth_extrin = color_stream_profile.get_extrinsics_to(depth_stream)

	pixel_coord = np.array(pixel_coord).tolist()
	depth_coord = rs.rs2_project_color_pixel_to_depth_pixel(
		depth_frame.get_data(), depth_scale,
		min_depth, max_depth,
		depth_intrin, color_intrin, depth_to_color_extrin, color_to_depth_extrin, pixel_coord)
	distance = depth_frame.get_distance(int(depth_coord[0]), int(depth_coord[1]))

	return np.array(rs.rs2_deproject_pixel_to_point(depth_intrin, depth_coord, distance)) / depth_scale

def back_to_pixel(point_coord, pipeline, old_origin_coord = [0, 0, 0], old_x_axis = [1, 0, 0], old_y_axis = [0, 1, 0], old_z_axis = None):
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
	M1 = np.column_stack((original_X, original_Y, original_Z)) # It's just the identity matrix
	p1 = np.matrix(coord).transpose()
	O1 = np.matrix([0, 0, 0]).transpose()


	O2 = np.matrix(new_origin_coord).transpose()
	new_X = np.matrix(new_x_axis_coord).transpose() - O2
	new_Y = np.matrix(new_y_axis_coord).transpose() - O2
	if new_z_axis_coord is not None:
		new_Z = np.matrix(new_z_axis_coord).transpose() - O2
	else:
		new_Z = compute_z_axis(new_X, new_Y)
	M2 = np.column_stack((new_X, new_Y, new_Z)) # It's just the identity matrix

	p2 = np.dot(np.linalg.inv(M2), ((O1 - O2) + np.dot(M1, p1))) # p2 = M2' * (O1 - O2 + M1 * p1)
	return p2

def compute_z_axis(x_axis, y_axis):
	z_axis = np.cross(x_axis.transpose(), y_axis.transpose()).transpose()
	# Getting a coherent Z norm
	norm_X = np.linalg.norm(x_axis)
	norm_Y = np.linalg.norm(y_axis)
	norm_Z = np.linalg.norm(z_axis)
	z_axis = (z_axis / norm_Z) * ((norm_X + norm_Y)/2)
	return z_axis


def draw_line(image, start, end, color = (0, 0, 0), thickness = 2):
	image = cv2.line(image, tuple(map(int, start)), tuple(map(int, end)), color = color, thickness = thickness)
	return image

def draw_cube(image, marker, depth_frame, pipeline):
	(top_left, top_right, bottom_right, bottom_left) = (marker[0][0][0], marker[0][0][1], marker[0][0][2], marker[0][0][3])

	top_left_3D    = get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
	top_right_3D   = get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
	bottom_left_3D = get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

	color_intrin = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

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


if __name__ == "__main__":
	# Example of usage: python convert_realsense_to_real_world.py --file recordings/aruco_code.bag
	import argparse

	parser = argparse.ArgumentParser(description='Transform the coordinate reference to fit an Aruco marker')
	parser.add_argument('-f', '--filename',	dest='filename', metavar='filename', type=str,
					help='Play a specific sequence .bag', required=False)
	args = parser.parse_args()
	filename = args.filename

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	pipeline = rs.pipeline()
	if filename is None:
		pipeline.start()
	else:
		pipeline = start_playback(filename, pipeline, realtime = True, repeat_playback = True)

	aruco_origin = [0, 0, 0]

	align_to = rs.stream.color
	align = rs.align(align_to)

	marker_size = 0.08
	while True:
		try:
			frames = pipeline.wait_for_frames()
		except:
			break

		# frames = align.process(frames)
		color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
		depth_frame = filters.aruco_detection_filters(frames.get_depth_frame())
		# depth_image = np.asanyarray(depth_frame.get_data()).copy()
		# depth_image = ((depth_image / depth_image.max()) * 255).astype(int)

		marked_frame, corners, ids = detectMarker(color_frame, aruco_dict)

		depth_image = np.array(rs.colorizer(2).colorize(depth_frame).get_data())
		if corners:
			# color_frame = draw_cube(color_frame, corners, depth_frame, pipeline)
			"""marker_corners = corners[0]
			(top_left, top_right, bottom_right, bottom_left) = (marker_corners[0][0], marker_corners[0][1], marker_corners[0][2], marker_corners[0][3])
			aruco_origin_3D = get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
			top_right_3D = get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
			bottom_left_3D = get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

			point = top_right_3D
			real_world_coord = translate_coord_to_world(coord = point,
											new_origin_coord = aruco_origin_3D,
											new_x_axis_coord = top_right_3D,
											new_y_axis_coord = bottom_left_3D,
											new_z_axis_coord = None)
			real_world_coord = real_world_coord.transpose() * marker_size"""

		cv2.imshow("marker", marked_frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if filename is None:
		pipeline = pipeline.stop()
	else:
		pipeline = stop_playback(pipeline)
