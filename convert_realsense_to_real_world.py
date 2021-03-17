import cv2
import numpy as np
import pyrealsense2 as rs

from recording import *

from cv2 import aruco

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
		new_Z = np.cross(new_X.transpose(), new_Y.transpose()).transpose()
		# Getting a coherent Z norm
		norm_X = np.linalg.norm(new_X)
		norm_Y = np.linalg.norm(new_Y)
		norm_Z = np.linalg.norm(new_Z)
		new_Z = (new_Z / norm_Z) * ((norm_X + norm_Y)/2)
	M2 = np.column_stack((new_X, new_Y, new_Z)) # It's just the identity matrix

	p2 = np.dot(np.linalg.inv(M2), ((O1 - O2) + np.dot(M1, p1))) # p2 = M2' * (O1 - O2 + M1 * p1)
	return p2

def draw_line(image, start, end, color = (0, 0, 0), thickness = 2):
	image = cv2.line(image, tuple(map(int, start)), tuple(map(int, end)), color = color, thickness = thickness)
	return image

def draw_cube(image, marker, depth_frame, pipeline):
	(top_left, top_right, bottom_right, bottom_left) = (marker[0][0][0], marker[0][0][1], marker[0][0][2], marker[0][0][3])

	top_left_3D    = get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
	top_right_3D   = get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
	bottom_left_3D = get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

	color_intrin = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

	p000 = rs.rs2_project_point_to_pixel(color_intrin, top_left_3D.tolist())#translate_coord_to_world(translate_coord_to_world([0, 0, 0], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0], [0, 0, 0], [1, 0, 0], [0, 1, 0]).transpose().tolist()[0])
	p001 = rs.rs2_project_point_to_pixel(color_intrin, top_right_3D.tolist())
	p010 = rs.rs2_project_point_to_pixel(color_intrin, bottom_left_3D.tolist())
	# p000 = translate_coord_to_world([0, 0, 0], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p000 = rs.rs2_project_point_to_pixel(color_intrin, p000)
    #
	# p001 = translate_coord_to_world([0, 0, 100], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p001 = rs.rs2_project_point_to_pixel(color_intrin, p001)
    #
	# p010 = translate_coord_to_world([0, 100, 0], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p010 = rs.rs2_project_point_to_pixel(color_intrin, p010)
    #
	# p011 = translate_coord_to_world([0, 100, 100], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p011 = rs.rs2_project_point_to_pixel(color_intrin, p011)
    #
	# p100 = translate_coord_to_world([100, 0, 0], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p100 = rs.rs2_project_point_to_pixel(color_intrin, p100)
    #
	# p101 = translate_coord_to_world([100, 0, 100], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p101 = rs.rs2_project_point_to_pixel(color_intrin, p101)
    #
	# p110 = translate_coord_to_world([100, 100, 0], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p110 = rs.rs2_project_point_to_pixel(color_intrin, p110)
    #
	# p111 = translate_coord_to_world([100, 100, 100], top_left_3D, top_right_3D, bottom_left_3D).transpose().tolist()[0]
	# p111 = rs.rs2_project_point_to_pixel(color_intrin, p111)

	# image = draw_line(image, p000, p001, color = (0, 0, 255)) # Blue
	# image = draw_line(image, p000, p010, color = (0, 255, 0)) # Green
	# image = draw_line(image, p000, p100, color = (255, 0, 0)) # Red
	# top_left_2D = rs.rs2_project_point_to_pixel(pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics(), top_left_3D.tolist())
	# image = draw_line(image, top_left_2D, top_right, (0, 0, 255))
	image = draw_line(image, p000, p001, (0, 0, 255))
	image = draw_line(image, p000, p010, (0, 255, 0))
	image = draw_line(image, p001, p010, (255, 0, 0))

	image = draw_line(image, top_left, top_right, (0, 0, 255), thickness = 1)
	image = draw_line(image, top_left, bottom_left, (0, 255, 0), thickness = 1)
	image = draw_line(image, top_right, bottom_left, (255, 0, 0), thickness = 1)
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

		frames = align.process(frames)
		color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
		depth_frame = frames.get_depth_frame()
		depth_image = np.asanyarray(depth_frame.get_data()).copy()
		# depth_image = ((depth_image / depth_image.max()) * 255).astype(int)

		marked_frame, corners, ids = detectMarker(color_frame, aruco_dict)

		depth_image = np.array(rs.colorizer(2).colorize(depth_frame).get_data())
		if corners:
			color_frame = draw_cube(color_frame, corners, depth_frame, pipeline)
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

		cv2.imshow("marker", color_frame)


		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if filename is None:
		pipeline = pipeline.stop()
	else:
		pipeline = stop_playback(pipeline)
