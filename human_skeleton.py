
import numpy as np
import scipy.spatial.transform
import math
from enum import Enum, auto
import cv2
import convert_realsense_to_real_world as coord_conversion


# /!\ Names of parts must match skeleton_tracker.cubemos_keypoints names
class BodyParts(Enum): # From the 18 keypoints of the coco dataset
	NOSE 			= auto()
	NECK 			= auto()
	RIGHT_SHOULDER 	= auto()
	RIGHT_ELBOW 	= auto()
	RIGHT_WRIST 	= auto()
	LEFT_SHOULDER 	= auto()
	LEFT_ELBOW 		= auto()
	LEFT_WRIST		= auto()
	RIGHT_HIP 		= auto()
	RIGHT_KNEE 		= auto()
	RIGHT_ANKLE 	= auto()
	LEFT_HIP 		= auto()
	LEFT_KNEE 		= auto()
	LEFT_ANKLE 		= auto()
	LEFT_EYE 		= auto()
	RIGHT_EYE 		= auto()
	RIGHT_EAR 		= auto()
	LEFT_EAR 		= auto()

class HumanSkeleton():
	def __init__(self):
		self.POSITION_NOT_FOUND2D = None #np.array([0, 0])
		self.POSITION_NOT_FOUND3D = None #np.array([0, 0, 0])
		self.body_parts_3D = {key.name : self.POSITION_NOT_FOUND3D for key in BodyParts}
		self.body_parts_2D = {key.name : self.POSITION_NOT_FOUND2D for key in BodyParts}

	def get_keypoint2D(self, key:BodyParts):
		if isinstance(key, str):
			key = BodyParts[key.upper()]
		return self.body_parts_2D[key.name]
	def get_keypoint3D(self, key:BodyParts):
		if isinstance(key, str):
			key = BodyParts[key.upper()]
		return self.body_parts_3D[key.name]

	def legs_overlapping(self):
		overlap = False
		# Check RIGHT_HIP-RIGHT_KNEE and LEFT_HIP-LEFT_KNEE
		overlap |= self.body_part_intersection(self.get_keypoint2D(BodyParts.RIGHT_HIP), self.get_keypoint2D(BodyParts.RIGHT_KNEE), self.get_keypoint2D(BodyParts.LEFT_HIP), self.get_keypoint2D(BodyParts.LEFT_KNEE), 20, 20)
		# Check RIGHT_HIP-RIGHT_KNEE and LEFT_KNEE-LEFT_ANKLE
		overlap |= self.body_part_intersection(self.get_keypoint2D(BodyParts.RIGHT_HIP), self.get_keypoint2D(BodyParts.RIGHT_KNEE), self.get_keypoint2D(BodyParts.LEFT_KNEE), self.get_keypoint2D(BodyParts.LEFT_ANKLE), 20, 20)
		# Check RIGHT_KNEE-RIGHT_ANKLE and LEFT_HIP-LEFT_KNEE
		overlap |= self.body_part_intersection(self.get_keypoint2D(BodyParts.RIGHT_KNEE), self.get_keypoint2D(BodyParts.RIGHT_ANKLE), self.get_keypoint2D(BodyParts.LEFT_HIP), self.get_keypoint2D(BodyParts.LEFT_KNEE), 20, 20)
		# Check RIGHT_KNEE-RIGHT_ANKLE and LEFT_KNEE-LEFT_ANKLE
		overlap |= self.body_part_intersection(self.get_keypoint2D(BodyParts.RIGHT_KNEE), self.get_keypoint2D(BodyParts.RIGHT_ANKLE), self.get_keypoint2D(BodyParts.LEFT_KNEE), self.get_keypoint2D(BodyParts.LEFT_ANKLE), 20, 20)
		return overlap

	def arms_overlapping(self):
		pass

	def body_position2D(self): # Take mean value between hips and shoulders with a weight of 3 for the hips (should fall on belly button)
		return ((self.body_parts_2D['LEFT_SHOULDER'] + self.body_parts_2D['RIGHT_SHOULDER'])*1 + (self.body_parts_2D['LEFT_HIP'] + self.body_parts_2D['RIGHT_HIP'])*3)/8
	def body_position3D(self): # Take mean value between hips and shoulders with a weight of 3 for the hips (should fall on belly button)
		return ((self.body_parts_3D['LEFT_SHOULDER'] + self.body_parts_3D['RIGHT_SHOULDER'])*1 + (self.body_parts_3D['LEFT_HIP'] + self.body_parts_3D['RIGHT_HIP'])*3)/8

	def from_tracker(self, tracker, id, depth_map = None):
		for key in self.body_parts_2D:
			if np.any(tracker.get_keypoint(key, person_id=id) == -1):
				self.body_parts_2D[key] = np.array([None, None])
				if tracker.is_3D_available and depth_map is not None:
					self.body_parts_3D[key] = np.array([None, None, None])
			else:
				self.body_parts_2D[key] = tracker.get_keypoint(key, person_id=id)
				if tracker.is_3D_available and depth_map is not None:
					self.body_parts_3D[key] = tracker.get_keypoint3D(key, depth_map, person_id=id)

	def from_json(self, data2D = None, data3D = None, specific_timestamp = None, id = None, depth_map = None): # id and depth_map unused, only for consistency with "from_tracker()"
		time_epsilon = 0.01
		if data2D is not None:
			for key in data2D:
				position = self.POSITION_NOT_FOUND2D

				if specific_timestamp is not None:
					possible_point = [np.array([point['x'], point['y']]) for point in data2D[key] if abs(point['time_ms'] - specific_timestamp) < time_epsilon]
					# print([abs(point['time_ms'] - specific_timestamp) < time_epsilon for point in data2D[key]])
					# print([len(str(round(point['time_ms']))) for point in data2D[key]])
					if len(possible_point) > 0:
						position = possible_point[0]
					else:
						position = self.POSITION_NOT_FOUND2D
				else:
					position = data2D[key]

				self.body_parts_2D[BodyParts[key].name] = position

		if data3D is not None:
			for key in data3D:
				position = self.POSITION_NOT_FOUND3D

				if specific_timestamp is not None:
					possible_point = [np.array([point['x'], point['y'], point['z']]) for point in data3D[key] if abs(point['time_ms'] - specific_timestamp) < time_epsilon]
					if len(possible_point) > 0:
						position = possible_point[0]
					else:
						position = self.POSITION_NOT_FOUND3D
				else:
					position = data3D[key]

				self.body_parts_3D[BodyParts[key].name] = position


	def to_json(self):
		coord2D = {key.name: self.get_keypoint2D(key) for key in BodyParts}
		coord3D = {key.name: self.get_keypoint3D(key) for key in BodyParts}
		return coord2D, coord3D


	def _get_line_intersection(self, start1, end1, start2, end2): # Coords in 2D
		# Get back to 2D if the coords are in higher dimension
		start1 = start1[:2]
		end1 = end1[:2]
		start2 = start2[:2]
		end2 = end2[:2]

		# Check if lines are vertical
		diffX1 = end1[0] - start1[0]
		diffX2 = end2[0] - start2[0]
		if diffX1 == 0:
			diffX1 = 0.0000001 # Avoid 0 by applying a value close to 0
		if diffX2 == 0:
			diffX2 = 0.0000001 # Avoid 0 by applying a value close to 0

		# With Ax = slope of line and Bx = ordonnée à l'origine
		A1, A2 = ((end1[1] - start1[1]) / diffX1), ((end2[1] - start2[1]) / diffX2)
		B1, B2 = end1[1] - A1 * end1[0], end2[1] - A2 * end2[0]

		if diffX1 == diffX2 and B1 != B2:
			return False

		# Intersection coord:
		intersectX = (B2 - B1) / (A1 - A2)
		intersectY = A1 * intersectX + B1


		if start1[0] <= intersectX <= end1[0]:
			return True
		else:
			return False

	def body_part_intersection(self, start1, end1, start2, end2, thickness1, thickness2):
		if start1 is None or start2 is None or end1 is None or end2 is None:
			return False
		# Get back to 2D if the coords are in higher dimension
		start1 = np.array(start1[:2])
		end1 = np.array(end1[:2])
		start2 = np.array(start2[:2])
		end2 = np.array(end2[:2])

		AB1 = end1 - start1
		AB2 = end2 - start2

		normalized_AB1 = AB1 / np.linalg.norm(AB1)
		normalized_AB2 = AB2 / np.linalg.norm(AB2)

		rotated_AB1 = np.array([normalized_AB1[1], normalized_AB1[0]])
		rotated_AB2 = np.array([normalized_AB2[1], normalized_AB2[0]])

		left_part1 = np.array([start1 - normalized_AB1 * thickness1 + rotated_AB1 * thickness1/2, end1 + normalized_AB1 * thickness1 + rotated_AB1 * thickness1/2])
		right_part1 = np.array([start1 - normalized_AB1 * thickness1 - rotated_AB1 * thickness1/2, end1 + normalized_AB1 * thickness1 - rotated_AB1 * thickness1/2])
		left_part2 = np.array([start2 - normalized_AB2 * thickness2 + rotated_AB2 * thickness2/2, end2 + normalized_AB2 * thickness2 + rotated_AB2 * thickness2/2])
		right_part2 = np.array([start2 - normalized_AB2 * thickness2 - rotated_AB2 * thickness2/2, end2 + normalized_AB2 * thickness2 - rotated_AB2 * thickness2/2])

		overlap = False
		overlap |= self._get_line_intersection(left_part1[0], left_part1[1], left_part2[0], left_part2[1])
		overlap |= self._get_line_intersection(left_part1[0], left_part1[1], right_part2[0], right_part2[1])
		overlap |= self._get_line_intersection(right_part1[0], right_part1[1], right_part2[0], right_part2[1])
		overlap |= self._get_line_intersection(right_part1[0], right_part1[1], left_part2[0], left_part2[1])
		return overlap

	def get_body_orientation(self):
		# Get the 4 vectors of the body (shoulders and hips) and normalize
		shoulder_to_shoulder_vector = None
		hip_to_hip_vector = None
		left_shoulder_to_hip_vector = None
		right_shoulder_to_hip_vector = None
		if self.get_keypoint3D("RIGHT_SHOULDER") is not self.POSITION_NOT_FOUND3D and self.get_keypoint3D("LEFT_SHOULDER") is not self.POSITION_NOT_FOUND3D and not np.array_equal(self.get_keypoint3D("RIGHT_SHOULDER"), self.get_keypoint3D("LEFT_SHOULDER")):
			shoulder_to_shoulder_vector = self.get_keypoint3D("RIGHT_SHOULDER") - self.get_keypoint3D("LEFT_SHOULDER")
			shoulder_to_shoulder_vector = shoulder_to_shoulder_vector / np.linalg.norm(shoulder_to_shoulder_vector)
		if self.get_keypoint3D("RIGHT_HIP") is not self.POSITION_NOT_FOUND3D and self.get_keypoint3D("LEFT_HIP") is not self.POSITION_NOT_FOUND3D and not np.array_equal(self.get_keypoint3D("RIGHT_HIP"), self.get_keypoint3D("LEFT_HIP")):
			hip_to_hip_vector = self.get_keypoint3D("RIGHT_HIP") - self.get_keypoint3D("LEFT_HIP")
			hip_to_hip_vector = hip_to_hip_vector / np.linalg.norm(hip_to_hip_vector)
		if self.get_keypoint3D("LEFT_SHOULDER") is not self.POSITION_NOT_FOUND3D and self.get_keypoint3D("LEFT_HIP") is not self.POSITION_NOT_FOUND3D and not np.array_equal(self.get_keypoint3D("LEFT_SHOULDER"), self.get_keypoint3D("LEFT_HIP")):
			left_shoulder_to_hip_vector = self.get_keypoint3D("LEFT_SHOULDER") - self.get_keypoint3D("LEFT_HIP")
			left_shoulder_to_hip_vector = left_shoulder_to_hip_vector / np.linalg.norm(left_shoulder_to_hip_vector)
		if self.get_keypoint3D("RIGHT_SHOULDER") is not self.POSITION_NOT_FOUND3D and self.get_keypoint3D("RIGHT_HIP") is not self.POSITION_NOT_FOUND3D and not np.array_equal(self.get_keypoint3D("RIGHT_SHOULDER"), self.get_keypoint3D("RIGHT_HIP")):
			right_shoulder_to_hip_vector = self.get_keypoint3D("RIGHT_SHOULDER") - self.get_keypoint3D("RIGHT_HIP")
			right_shoulder_to_hip_vector = right_shoulder_to_hip_vector / np.linalg.norm(right_shoulder_to_hip_vector)


		body_orientation = []
		if left_shoulder_to_hip_vector is not None and shoulder_to_shoulder_vector is not None:
			body_orientation.append(np.cross(left_shoulder_to_hip_vector, shoulder_to_shoulder_vector))
		if right_shoulder_to_hip_vector is not None and shoulder_to_shoulder_vector is not None:
			body_orientation.append(np.cross(right_shoulder_to_hip_vector, shoulder_to_shoulder_vector))
		if left_shoulder_to_hip_vector is not None and hip_to_hip_vector is not None:
			body_orientation.append(np.cross(left_shoulder_to_hip_vector, hip_to_hip_vector))
		if right_shoulder_to_hip_vector is not None and hip_to_hip_vector is not None:
			body_orientation.append(np.cross(right_shoulder_to_hip_vector, hip_to_hip_vector))

		if len(body_orientation) == 0:
			return None
		else:
			mean = np.mean(np.array(body_orientation), axis = 0)
			mean = mean / np.linalg.norm(mean)
			return mean

	def get_center_of_gravity(self):
		body = []
		if self.get_keypoint3D("RIGHT_SHOULDER") is not self.POSITION_NOT_FOUND3D:
			body.append(self.get_keypoint3D("RIGHT_SHOULDER"))
		if self.get_keypoint3D("LEFT_SHOULDER") is not self.POSITION_NOT_FOUND3D:
			body.append(self.get_keypoint3D("LEFT_SHOULDER"))
		if self.get_keypoint3D("RIGHT_HIP") is not self.POSITION_NOT_FOUND3D:
			body.append(self.get_keypoint3D("RIGHT_HIP"))
		if self.get_keypoint3D("LEFT_HIP") is not self.POSITION_NOT_FOUND3D:
			body.append(self.get_keypoint3D("LEFT_HIP"))

		if len(body) == 0:
			return self.POSITION_NOT_FOUND3D

		else:
			return np.mean(np.array(body), axis = 0)

	def get_center_of_gravity_2d(self):
		body = []
		if self.get_keypoint2D("RIGHT_SHOULDER") is not self.POSITION_NOT_FOUND2D:
			body.append(self.get_keypoint2D("RIGHT_SHOULDER"))
		if self.get_keypoint2D("LEFT_SHOULDER") is not self.POSITION_NOT_FOUND2D:
			body.append(self.get_keypoint2D("LEFT_SHOULDER"))
		if self.get_keypoint2D("RIGHT_HIP") is not self.POSITION_NOT_FOUND2D:
			body.append(self.get_keypoint2D("RIGHT_HIP"))
		if self.get_keypoint2D("LEFT_HIP") is not self.POSITION_NOT_FOUND2D:
			body.append(self.get_keypoint2D("LEFT_HIP"))

		if len(body) == 0:
			return self.POSITION_NOT_FOUND2D

		else:
			return np.mean(np.array(body), axis = 0)

	def get_face_location(self):
		face_location = None
		if self.check_keypoints(["NECK", "NOSE", "LEFT_EAR", "RIGHT_EAR"]):
			bottom = self.get_keypoint2D("NECK")[1]
			mid = self.get_keypoint2D("NOSE")[1]
			top = mid - (bottom - mid)
			left = self.get_keypoint2D("LEFT_EAR")[0]
			right = self.get_keypoint2D("RIGHT_EAR")[0]
			if right < left:
				tmp = right
				right = left
				left = tmp
			if bottom < top:
				tmp = bottom
				bottom = top
				top = tmp
			distance_h = right - left
			right += distance_h * 0.1
			left -= distance_h * 0.1
			distance_v = bottom - top
			bottom += distance_v * -0.1
			top -= distance_v * -0.1

			right = int(right)
			left = int(left)
			bottom = int(bottom)
			top = int(top)
			if top >= 0 and right >= 0 and bottom >= 0 and left >= 0:
				face_location = (top, right, bottom, left)
		return face_location

	def get_bounding_box_3d(self):
		try:
			min_x = min([self.get_keypoint3D(key)[0] for key in BodyParts if self.get_keypoint3D(key) is not self.POSITION_NOT_FOUND3D])
			max_x = max([self.get_keypoint3D(key)[0] for key in BodyParts if self.get_keypoint3D(key) is not self.POSITION_NOT_FOUND3D])
			min_y = min([self.get_keypoint3D(key)[1] for key in BodyParts if self.get_keypoint3D(key) is not self.POSITION_NOT_FOUND3D])
			max_y = max([self.get_keypoint3D(key)[1] for key in BodyParts if self.get_keypoint3D(key) is not self.POSITION_NOT_FOUND3D])
			min_z = min([self.get_keypoint3D(key)[2] for key in BodyParts if self.get_keypoint3D(key) is not self.POSITION_NOT_FOUND3D])
			max_z = max([self.get_keypoint3D(key)[2] for key in BodyParts if self.get_keypoint3D(key) is not self.POSITION_NOT_FOUND3D])

			return [np.array(coord) for coord in [
				[min_x, min_y, min_z],
				[min_x, min_y, max_z],
				[min_x, max_y, min_z],
				[min_x, max_y, max_z],
				[max_x, min_y, min_z],
				[max_x, min_y, max_z],
				[max_x, max_y, min_z],
				[max_x, max_y, max_z],
			]]
		except ValueError:
			return None

	def rotate_bounding_box_with_body_orientation(self, increase_body_height = False):
		if self.get_bounding_box_3d() is None:
			return None
		body_orientation = self.get_body_orientation()
		if body_orientation is None:
			return None
		height_multiplier = increase_body_height
		if isinstance(height_multiplier, bool):
			if height_multiplier:
				height_multiplier = 1.0
			else:
				height_multiplier = 1.1
		mean_pos = self.get_center_of_gravity()
		rotation = scipy.spatial.transform.Rotation.from_rotvec([0, body_orientation[0], 0]) # Only take care of Y-axis
		rotation_matrix = rotation.as_matrix()
		new_coords = []
		for coord in self.get_bounding_box_3d():
			new_coord = coord - mean_pos
			new_coord = np.dot(new_coord, rotation_matrix)
			new_coord[1] *= height_multiplier # Increase Y-axis
			new_coord += mean_pos
			new_coords.append(new_coord)

		return new_coords

	def display_BBox(self, frame, pipeline, color = (0, 0, 0), display_front = True, display_back = True):
		body_BBox = self.rotate_bounding_box_with_body_orientation(increase_body_height = True)
		# body_pos = self.get_center_of_gravity()
		min_z = np.min(body_BBox, axis = 0)
		max_z = np.max(body_BBox, axis = 0)
		if body_BBox is not None:
			display_coord = [True] * len(body_BBox)
			for i in range(len(display_coord)):
				# if (not display_front and body_BBox[i][2] < body_pos[2]) or (not display_back and body_BBox[i][2] > body_pos[2]):
				if (not display_front and body_BBox[i][2] == min_z[2]) or (not display_back and body_BBox[i][2] == max_z[2]):
					display_coord[i] = False
			p000, p001, p010, p011, p100, p101, p110, p111 = list(map(lambda coord: tuple(map(int, coord_conversion.back_to_pixel(coord, pipeline))), body_BBox))
			if display_coord[0] and display_coord[1]:
				frame = cv2.line(frame, p000, p001, color, 2)
			if display_coord[0] and display_coord[2]:
				frame = cv2.line(frame, p000, p010, color, 2)
			if display_coord[0] and display_coord[4]:
				frame = cv2.line(frame, p000, p100, color, 2)
			if display_coord[2] and display_coord[6]:
				frame = cv2.line(frame, p010, p110, color, 2)
			if display_coord[2] and display_coord[3]:
				frame = cv2.line(frame, p010, p011, color, 2)
			if display_coord[3] and display_coord[7]:
				frame = cv2.line(frame, p011, p111, color, 2)
			if display_coord[3] and display_coord[1]:
				frame = cv2.line(frame, p011, p001, color, 2)
			if display_coord[4] and display_coord[6]:
				frame = cv2.line(frame, p100, p110, color, 2)
			if display_coord[4] and display_coord[5]:
				frame = cv2.line(frame, p100, p101, color, 2)
			if display_coord[5] and display_coord[7]:
				frame = cv2.line(frame, p101, p111, color, 2)
			if display_coord[5] and display_coord[1]:
				frame = cv2.line(frame, p101, p001, color, 2)
			if display_coord[6] and display_coord[7]:
				frame = cv2.line(frame, p110, p111, color, 2)
		return frame

	def check_keypoints(self, names):
		for name in names:
			if self.get_keypoint2D(name) is self.POSITION_NOT_FOUND2D:
				return False
		return True

