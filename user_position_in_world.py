import cv2
import imutils
import numpy as np
import math
import pyrealsense2 as rs
import time
#from skeletontracker import skeletontracker
#import util as cm # Utils from the skeleton sdk

import skeleton_tracker as st
import human_skeleton

from recording import *
from convert_realsense_to_real_world import *

from cv2 import aruco

import automatic_capture as auto_cap

import InfluxDB_connection

class Map:
	def __init__(self, width_m = 3, height_m = 3, center_x = None, center_y = None, width_px = 600, height_px = 600, marker_size = 1.0, marker_position = "wall"):
		self.width_m = width_m
		self.width_px = width_px
		self.height_m = height_m
		self.height_px = height_px
		self.scale_x = width_px / width_m
		self.scale_y = height_px / height_m
		self.scale = min(self.scale_x, self.scale_y)
		self.center_x = center_x if center_x is not None else width_m/2
		self.center_y = center_y if center_y is not None else height_m/2

		self.reference_2D = None	# Position of the marker in the frame
		self.reference_3D = None	# Position of the marker in 3D (from the camera)

		self.users_position_2D = [] 		# Position of the users in the frame
		self.users_position_real_world = [] # Position of the users in the real world
		self.users_names = []
		self.users_colors = []

		self.user_walk_tracking = {}
		self.user_walk_tracking_name = {}

		self.user_orientation = []
		self.skeletons = []
		self.previous_skeletons = []

		self.image = None

		self.marker_size = marker_size
		self.marker_position = marker_position

		self.display_axes = [0, 2, 1]
		if marker_position == "ground":
			self.display_axes = [0, 1, 2]

		self.right_footstep_img = cv2.imread("footsteps/right_foot.png", cv2.IMREAD_UNCHANGED)
		self.left_footstep_img = cv2.imread("footsteps/left_foot.png", cv2.IMREAD_UNCHANGED)

	def update_reference_coord(self, marker_corners, depth_frame, pipeline):
		(top_left, top_right, bottom_right, bottom_left) = (marker_corners[0][0], marker_corners[0][1], marker_corners[0][2], marker_corners[0][3])
		aruco_origin_3D = get_3D_coord_from_realsense(top_left.tolist(), depth_frame, pipeline)
		top_right_3D = get_3D_coord_from_realsense(top_right.tolist(), depth_frame, pipeline)
		bottom_left_3D = get_3D_coord_from_realsense(bottom_left.tolist(), depth_frame, pipeline)

		if np.isnan(aruco_origin_3D).any() or np.isnan(top_right_3D).any() or np.isnan(bottom_left_3D).any():
			return None
		else:
			self.reference_2D = {'origin' : top_left, 			'x_axis' : top_right, 		'y_axis' : bottom_left}
			self.reference_3D = {'origin' : aruco_origin_3D, 	'x_axis' : top_right_3D, 	'y_axis' : bottom_left_3D}
			return self.reference_3D

	def add_users(self, skeletons, depth_frame, pipeline, marker_corners = None, keypoint="NECK", name = None, previous = None, limit = None, cross_color = (0, 0, 0)):
		# self.users_position_real_world = []
		# self.users_position_2D = []
		position_real_world = np.array([0, 0, 0])
		# if marker_corners is not None:
		# 	self.update_reference_coord(marker_corners, depth_frame, pipeline)
		self.reference_3D = {'origin': [0, 0, 0], 'x_axis': [1, 0, 0], 'y_axis': [0, 0, 1]}
		if self.reference_3D is not None: # If a marker has been detected previously
			for id_skel, skeleton in enumerate(skeletons['body']):
				user_position = get_user_position_from_skeleton(skeleton, keypoint = keypoint)
				if user_position is not None:
					position_3D = get_3D_coord_from_realsense(user_position, depth_frame, pipeline)
					position_real_world = position_3D
					# position_real_world = translate_coord_to_world(coord = position_3D,
					# 							new_origin_coord = self.reference_3D['origin'],
					# 							new_x_axis_coord = self.reference_3D['x_axis'],
					# 							new_y_axis_coord = self.reference_3D['y_axis'],
					# 							new_z_axis_coord = None) * self.marker_size
					if (previous is None) or (limit is None) or (np.linalg.norm(position_real_world - np.array(previous)) < limit):
						self.users_position_real_world.append((len(self.users_names), position_real_world))
						self.users_position_2D.append((len(self.users_names), user_position))
						self.users_names.append((len(self.users_names), name))
						self.users_colors.append(cross_color)
					# self.users_position_real_world.append((skeletons['ids'][id_skel], position_real_world))
					# self.users_position_2D.append((skeletons['ids'][id_skel], user_position))
					# self.users_names.append((skeletons['ids'][id_skel], name))
		return position_real_world

	def add_user_from_json(self, skeleton2D_and_3D, depth_frame = None, pipeline = None, marker_corners = None, keypoint="NECK", name = None, previous = None, limit = None, cross_color = (0, 0, 0), specific_timestamp = None):
		skeleton2D, skeleton3D = skeleton2D_and_3D
		if specific_timestamp is None:
			if depth_frame is None:
				raise Exception("Provide a timestamp or at least a depth frame to add the position of a user")
			else:
				specific_timestamp = depth_frame.timestamp/1000

		# if marker_corners is not None:
		# 	self.update_reference_coord(marker_corners, depth_frame, pipeline)

		human = human_skeleton.HumanSkeleton()
		human.from_json(data2D = skeleton2D, data3D = skeleton3D, specific_timestamp = specific_timestamp)
		position_real_world = np.array([0, 0, 0])
		if human.get_keypoint3D(keypoint) is not None:
			user_position = human.get_keypoint2D(keypoint)
			position_3D = human.get_keypoint3D(keypoint)
			position_real_world = position_3D
			self.user_orientation.append((len(self.users_names), human.get_body_orientation()))
			# position_real_world = translate_coord_to_world(coord = position_3D,
			# 							new_origin_coord = self.reference_3D['origin'],
			# 							new_x_axis_coord = self.reference_3D['x_axis'],
			# 							new_y_axis_coord = self.reference_3D['y_axis'],
			# 							new_z_axis_coord = None) * self.marker_size

			if (previous is None) or (limit is None) or (np.linalg.norm(position_real_world - np.array(previous)) < limit):
				self.users_position_real_world.append((len(self.users_names), position_real_world))
				self.users_position_2D.append((len(self.users_names), user_position))
				self.users_names.append((len(self.users_names), name))
				self.users_colors.append(cross_color)
		return position_real_world

	def add_footsteps_from_json(self, skeleton2D_and_3D, depth_frame = None, pipeline = None, marker_corners = None, keypoint="NECK", names = [], previous = None, limit = None, cross_color = (0, 0, 0), specific_timestamp = None):
		skeleton2D, skeleton3D = skeleton2D_and_3D
		if specific_timestamp is None:
			if depth_frame is None:
				raise Exception("Provide a timestamp or at least a depth frame to add the position of a user")
			else:
				specific_timestamp = depth_frame.timestamp/1000
		# if marker_corners is not None:
		# 	self.update_reference_coord(marker_corners, depth_frame, pipeline)

		for skeleton_id in range(len(skeleton2D)):
			human = human_skeleton.HumanSkeleton()
			human.from_json(data2D = skeleton2D[skeleton_id], data3D = skeleton3D[skeleton_id], specific_timestamp = specific_timestamp)
			# position_real_world = np.array([0, 0, 0])
			# if human.get_keypoint3D(keypoint) is not None:
			# 	right_foot_2D = human.get_keypoint2D("RIGHT_ANKLE")
			# 	right_foot_3D = human.get_keypoint3D("RIGHT_ANKLE")
			# 	right_foot_real_world = position_3D
			# 	left_foot_2D = human.get_keypoint2D("LEFT_ANKLE")
			# 	left_foot_3D = human.get_keypoint3D("LEFT_ANKLE")
			# 	left_foot_real_world = position_3D
			# 	self.user_orientation.append((len(self.users_names), human.get_body_orientation()))
			# 	# position_real_world = translate_coord_to_world(coord = position_3D,
			# 	# 							new_origin_coord = self.reference_3D['origin'],
			# 	# 							new_x_axis_coord = self.reference_3D['x_axis'],
			# 	# 							new_y_axis_coord = self.reference_3D['y_axis'],
			# 	# 							new_z_axis_coord = None) * self.marker_size
	        #
			# 	self.users_positions_real_world.append((len(self.users_names), position_real_world))
			# 	self.users_positions_2D.append((len(self.users_names), user_position))
			self.skeletons.append((len(self.users_names), human))
			self.users_names.append((len(self.users_names), names[skeleton_id]))
			self.users_colors.append(cross_color)

	# def add_user(self, skeletons, depth_frame, pipeline, marker_corners = None, keypoint=None):

	def display_map(self, bg = (255, 255, 255)):
		self.image = self._init_map(bg = bg)
		self.image = self._add_center_reference()
		self.image = self._display_users(cross_size = 6)
		self.image = self.display_walk_tracking(color = (0, 255, 0), number_of_frames_to_average = 10)
		self.image = self._display_footsteps()
		return self.image


	def _init_map(self, bg = (0, 0, 0)):
		self.image = np.zeros((self.height_px, self.width_px, 3), np.uint8)
		self.image[:, :] = bg
		return self.image

	def _draw_arrow(self, image, start, end, color = (255, 255, 255), thickness = 2):
		# Main stroke
		image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), color, thickness)

		# Head
		head_angle_deg = math.radians(30)
		head_size = 0.2
		# left
		head_to_tail = np.array([end[0] - start[0], end[1] - start[1]]) * head_size
		head_1_x = head_to_tail[0] * math.cos(head_angle_deg) - head_to_tail[1] * math.sin(head_angle_deg)
		head_1_y = head_to_tail[0] * math.sin(head_angle_deg) + head_to_tail[1] * math.cos(head_angle_deg)
		# right
		head_2_x = head_to_tail[0] * math.cos(-head_angle_deg) - head_to_tail[1] * math.sin(-head_angle_deg)
		head_2_y = head_to_tail[0] * math.sin(-head_angle_deg) + head_to_tail[1] * math.cos(-head_angle_deg)

		image = cv2.line(image, (int(end[0]), int(end[1])), (int(end[0] - head_1_x), int(end[1] - head_1_y)), color, thickness)
		image = cv2.line(image, (int(end[0]), int(end[1])), (int(end[0] - head_2_x), int(end[1] - head_2_y)), color, thickness)
		return image

	def _draw_cross(self, image, position, cross_size = 3, color = (0, 0, 0), thickness = 2):
		image = cv2.line(image, (int(position[0] - cross_size/2), int(position[1] - cross_size/2)), (int(position[0] + cross_size/2), int(position[1] + cross_size/2)), color, thickness)
		image = cv2.line(image, (int(position[0] - cross_size/2), int(position[1] + cross_size/2)), (int(position[0] + cross_size/2), int(position[1] - cross_size/2)), color, thickness)
		return image


	def _add_center_reference(self):
		if self.reference_2D is not None:
			ref_pos_x = self.center_x * self.scale
			ref_pos_y = self.center_y * self.scale
			self.image = self._draw_arrow(self.image, (ref_pos_x, ref_pos_y), (ref_pos_x + self.scale, ref_pos_y), color=(0, 0, 255))#cv2.line(self.image, (int(ref_pos_x), int(ref_pos_y)), (int(ref_pos_x + 1 * self.scale), int(ref_pos_y)), (0, 0, 255), 2) # X
			self.image = self._draw_arrow(self.image, (ref_pos_x, ref_pos_y), (ref_pos_x, ref_pos_y + self.scale), color=(255, 0, 0))# self.image = cv2.line(self.image, (int(ref_pos_x), int(ref_pos_y)), (int(ref_pos_x), int(ref_pos_y - 1 * self.scale)), (255, 0, 0), 2) # Y
		else:
			pass
		return self.image

	def _display_users(self, cross_size = 3):
		for i, (id, user_pos_3D) in enumerate(self.users_position_real_world):
			color = self.users_colors[i]
			position_x = (user_pos_3D[self.display_axes[0]] + self.center_x) * self.scale
			position_y = (user_pos_3D[self.display_axes[1]] + self.center_y) * self.scale

			orientation = self.user_orientation[i][1] * 100
			orientation_x = orientation[self.display_axes[0]]
			orientation_y = orientation[self.display_axes[1]]
			# Z axis skipped for the projection
			self.image = self._draw_cross(self.image, [position_x, position_y], cross_size = cross_size, color = color, thickness = 2)
			self.image = self._draw_arrow(self.image, [position_x, position_y], [position_x + orientation_x, position_y + orientation_y], color = (255, 0, 0))


			user_name = [name for i, name in self.users_names if i == id]
			if len(user_name) > 0:
				self.image = cv2.putText(self.image, user_name[0], (int(position_x + 10), int(position_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
		return self.image

	def display_walk_tracking(self, number_of_frames_to_average = 10, color = (0, 255, 255)):
		for i in range(len(self.user_walk_tracking_name)):
			_number_of_frames_to_average = min(len(self.user_walk_tracking[i]), number_of_frames_to_average)
			position_2D = np.array([self.user_walk_tracking[i][-1][self.display_axes[0]], self.user_walk_tracking[i][-1][self.display_axes[1]]])
			mean_pos_x = np.mean(np.array(self.user_walk_tracking[i])[-_number_of_frames_to_average:, self.display_axes[0]])
			mean_pos_y = np.mean(np.array(self.user_walk_tracking[i])[-_number_of_frames_to_average:, self.display_axes[1]])
			walked_2D = np.array([mean_pos_x, mean_pos_y]) - position_2D

			position_x = (position_2D[0] + self.center_x) * self.scale
			position_y = (position_2D[1] + self.center_y) * self.scale

			walk_x = walked_2D[0] * self.scale
			walk_y = walked_2D[1] * self.scale
			self.image = self._draw_arrow(self.image, [position_x, position_y], [position_x - walk_x * (10/_number_of_frames_to_average), position_y - walk_y * (10/_number_of_frames_to_average)], color, 3)


			fps = 15
			distance = np.linalg.norm(walked_2D)
			frames_considered = _number_of_frames_to_average/2
			speed = distance / (fps*frames_considered)
			self.image = cv2.putText(self.image, str(round(speed, 2)) + "m/s", (int(position_x - walk_x * (10/_number_of_frames_to_average) + 10), int(position_y - walk_y * (10/_number_of_frames_to_average) + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
		return self.image

	def add_walk_tracking(self, skeleton2D_and_3D, depth_frame = None, pipeline = None, marker_corners = None, keypoint="NECK", name = None, specific_timestamp = None, id_person = 0):
		skeleton2D, skeleton3D = skeleton2D_and_3D
		if specific_timestamp is None:
			if depth_frame is None:
				raise Exception("Provide a timestamp or at least a depth frame to add the position of a user")
			else:
				specific_timestamp = depth_frame.timestamp/1000

		human = human_skeleton.HumanSkeleton()
		human.from_json(data2D = skeleton2D, data3D = skeleton3D, specific_timestamp = specific_timestamp)
		position_real_world = np.array([0, 0, 0])
		if human.get_keypoint3D(keypoint) is not None:
			user_position = human.get_keypoint2D(keypoint)
			position_3D = human.get_keypoint3D(keypoint)
			position_real_world = position_3D
			# position_real_world = translate_coord_to_world(coord = position_3D,
			# 							new_origin_coord = self.reference_3D['origin'],
			# 							new_x_axis_coord = self.reference_3D['x_axis'],
			# 							new_y_axis_coord = self.reference_3D['y_axis'],
			# 							new_z_axis_coord = None) * self.marker_size
			if id_person not in self.user_walk_tracking:
				self.user_walk_tracking[id_person] = []
				self.user_walk_tracking_name[id_person] = []
			self.user_walk_tracking[id_person].append(position_real_world)
			self.user_walk_tracking_name[id_person].append(name)
		return position_real_world

	def _display_footsteps(self):
		for i, (id, skeleton) in enumerate(self.skeletons):
			color = self.users_colors[i]
			right_foot = skeleton.get_keypoint3D('RIGHT_ANKLE')
			orientation = skeleton.get_body_orientation()
			orientation_degrees = 0.0
			if orientation is not None:
				orientation = orientation[self.display_axes[:2]]
				orientation_degrees = math.degrees(np.arccos(np.dot(orientation, np.array([-1, 0])))) * (-1 if np.dot(orientation, np.array([0, -1])) < 0 else 1) - 90
			if right_foot is not None:
				right_foot = (right_foot[self.display_axes[:2]] + np.array([self.center_x, self.center_y])) * self.scale
				self.image = self._add_image(self.image, self.right_footstep_img, right_foot, angle = orientation_degrees)
			left_foot = skeleton.get_keypoint3D('LEFT_ANKLE')
			if left_foot is not None:
				left_foot = (left_foot[self.display_axes[:2]] + np.array([self.center_x, self.center_y])) * self.scale
				self.image = self._add_image(self.image, self.left_footstep_img, left_foot, angle = orientation_degrees)

			if orientation is not None:
				neck = skeleton.get_keypoint3D('NECK')
				if neck is not None:
					neck = (neck[self.display_axes[:2]] + np.array([self.center_x, self.center_y])) * self.scale
					print(np.dot(orientation, np.array([0, -1])), orientation_degrees)
					self.image = self._draw_arrow(self.image, neck, neck + orientation * 100, color=(0, 255, 255))

					if len(self.previous_skeletons) > 0 and skeleton.get_center_of_gravity() is not None and self.previous_skeletons[i][1].get_center_of_gravity() is not None:
						self.image = cv2.putText(self.image, str(round(np.linalg.norm(skeleton.get_center_of_gravity() - self.previous_skeletons[i][1].get_center_of_gravity())/15, 2)) + "m/s", (int(neck[0] + 5), int(neck[1] + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255))

			# self.image = self._draw_arrow(self.image, [position_x, position_y], [position_x + orientation_x, position_y + orientation_y], color = (255, 0, 0))


			# user_name = [name for i, name in self.users_names if i == id]
			# if len(user_name) > 0:
			# 	self.image = cv2.putText(self.image, user_name[0], (int(position_x + 10), int(position_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
		return self.image

	def _add_image(self, image, new_image, new_image_coord, alpha = 1.0, image_position = "center", angle = 0.0):
		# new_image_coord = np.array(new_image_coord) # Make a copy
		image_rotated = imutils.rotate_bound(new_image, angle)
		new_image_coord = np.array([new_image_coord[1], new_image_coord[0]])
		if image_position == "center":
			new_image_coord -= np.array(image_rotated.shape)[:2] / 2

		left   = int((new_image_coord[1]))
		top    = int((new_image_coord[0]))
		bottom = int((new_image_coord[0] + image_rotated.shape[0]))
		right  = int((new_image_coord[1] + image_rotated.shape[1]))

		image[top:bottom, left:right] = (image_rotated[:,:, :3] * alpha * (image_rotated[:,:,3, None] / 255)) + (image[top:bottom, left:right] * (1- (alpha * (image_rotated[:,:,3, None] / 255))))
		# print(image[top:bottom, left:right])s
		return image


	def clear_users(self):
		self.users_position_real_world = []
		self.users_position_2D = []
		self.users_names = []
		self.user_orientation = []
		self.previous_skeletons = self.skeletons
		self.skeletons = []


def get_user_position_from_skeleton(skeleton, keypoint = None):
	skeleton = np.array(skeleton)
	if keypoint is None:
		Xs = skeleton[:, 0]
		Ys = skeleton[:, 1]
		mean = np.array([np.mean(Xs[Xs > 0]), np.mean(Ys[Ys > 0])])
		return mean
	else:
		return skeleton[st.cubemos_keypoints[keypoint].value]

if __name__ == "__main__":

	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	pipeline = rs.pipeline()

	aruco_origin = [0, 0, 0]
	marker_size = 0.08

	# skeletrack = st.SkeletonTracker(choose_best_module=True)
	joint_confidence = 0.2
	# pipeline.start()
	skeleton_db2D_and_skeleton_db3D = InfluxDB_connection.ListicClientDB(server_side = False).retrieve_skeleton_keypoints("recordings/walking_normal.bag")
	pipeline = start_playback("recordings/walking_normal.bag", pipeline = pipeline, realtime = True, repeat_playback = True)
	# automatic_capturer = auto_cap.AutomaticCapture(enable = False, tracker = skeletrack, pipeline = pipeline, recording_dir = "test_record", latency = 10)

	marker_corners = None

	# fake_corners = [
	# 	[
	# 		[
	# 			np.array([24, 541]),
	# 			np.array([156, 537]),
	# 			np.array([157, 415]),
	# 			np.array([34, 417]),
	# 		]
	# 	]
	# ]

	maraudeur = Map(7000, 7000, center_x = 3500, center_y = 0, marker_size = marker_size, marker_position = "camera") # marker_position = {wall | ground | camera}

	previous_right = [0, 0, 0]
	previous_left = [0, 0, 0]

	playing = True
	while True:
		# This call waits until a new coherent set of frames is available on a device
		# Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
		if playing:
			# start_time = time.time()
			frames = pipeline.wait_for_frames()
			frames.keep()
			#frames = align.process(frames)
			color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
			depth_frame = frames.get_depth_frame()

			# skeletons = skeletrack.track_skeletons(color_frame, debug_time = False)  # "Live" solution
			skeletons = skeleton_db2D_and_skeleton_db3D		# "InfluxDB" solution




			marked_frame, corners, ids = detectMarker(color_frame, aruco_dict)


			if corners:
				marker_corners = corners[0]

			# maraudeur.clear_users()
			# previous_right = maraudeur.add_users(skeletons, depth_frame, pipeline, marker_corners, keypoint="RIGHT_ANKLE", name = "Marc", previous = previous_right, limit = 150, cross_color=(255, 0, 0))
			# previous_left = maraudeur.add_users(skeletons, depth_frame, pipeline, marker_corners, keypoint="LEFT_ANKLE", name = "Marc", previous = previous_left, limit = 150, cross_color=(0, 0, 255))
            #
			# for keypoint in human_skeleton.BodyParts:
			# 	if keypoint.name in ["RIGHT_ANKLE", "LEFT_ANKLE"]:
			# 		maraudeur.add_users(skeletons, depth_frame, pipeline, marker_corners, keypoint=keypoint.name, name = "Marc")
			maraudeur.clear_users()
			maraudeur.add_footsteps_from_json(skeletons, names = ["Person" + str(i+1) for i in range(len(skeletons[0]))], depth_frame = depth_frame)
			# previous_right = maraudeur.add_user_from_json(skeletons, depth_frame, pipeline, marker_corners, keypoint="RIGHT_ANKLE", name = "Marc", previous = previous_right, limit = 1000, cross_color=(255, 0, 0))
			# previous_left = maraudeur.add_user_from_json(skeletons, depth_frame, pipeline, marker_corners, keypoint="LEFT_ANKLE", name = "Marc", previous = previous_left, limit = 1000, cross_color=(0, 0, 255))
			# maraudeur.add_walk_tracking(skeletons, depth_frame, pipeline, marker_corners, keypoint="NECK", name = "Marc")

			cv2.imshow("Maraudeur", maraudeur.display_map())
			cv2.imshow("Camera", cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
			# cv2.imshow("Camera", cv2.cvtColor(skeletrack.draw_skeleton(color_frame), cv2.COLOR_BGR2RGB))
			# print("Total     time : ", round((time.time() - start_time), 3))



		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			pipeline = stop_playback(pipeline)
			break
		if key == ord('p'):
			if playing:
				pipeline = pause_playback(pipeline)
			else:
				pipeline = resume_playback(pipeline)
			playing = not playing
