modules_importance = [
	('openpose', 20),
	('cubemos' , 90),
	('trt_pose', 80),
]
skeleton_tracking_modules = []
try:
	import openpose_python as op
	skeleton_tracking_modules.append('openpose')
except Exception as e:
	# raise e
	pass

try:
	import os
	if 'LD_LIBRARY_PATH' not in os.environ:
		os.environ['LD_LIBRARY_PATH'] = '/usr/lib64/'
	from skeletontracker import skeletontracker
	skeleton_tracking_modules.append('cubemos')
except Exception as e:
	# raise e
	pass

try:
	import trt_pose
	import trt_pose.coco
	import trt_pose.models
	try:
		import json
		import torch
		import torch2trt
		import torchvision.transforms as transforms
		import PIL.Image
		skeleton_tracking_modules.append('trt_pose')
	except Exception as e:
		print("Importation of TRT-pose was ok, but there is additionnal packages raising errors :", e)
except Exception as e:
	# raise e
	pass

import os
import threading
import time

import cv2
import numpy as np

available_modules_importance = [module for module in modules_importance if module[0] in skeleton_tracking_modules]
unavailable_modules_importance = [module for module in modules_importance if module[0] not in skeleton_tracking_modules]

trt_pose_data_dir = 'trt_pose_data'

from enum import Enum, unique
@unique
class cubemos_keypoints(Enum):
	NOSE 			= 0
	NECK 			= 1
	RIGHT_SHOULDER 	= 2
	RIGHT_ELBOW 	= 3
	RIGHT_WRIST 	= 4
	LEFT_SHOULDER 	= 5
	LEFT_ELBOW 		= 6
	LEFT_WRIST		= 7
	RIGHT_HIP 		= 8
	RIGHT_KNEE 		= 9
	RIGHT_ANKLE 	= 10
	LEFT_HIP 		= 11
	LEFT_KNEE 		= 12
	LEFT_ANKLE 		= 13
	LEFT_EYE 		= 14
	RIGHT_EYE 		= 15
	LEFT_EAR 		= 16
	RIGHT_EAR 		= 17

@unique
class openpose_keypoints(Enum):
	NOSE 			= 0
	NECK 			= 1
	RIGHT_SHOULDER 	= 2
	RIGHT_ELBOW 	= 3
	RIGHT_WRIST 	= 4
	LEFT_SHOULDER 	= 5
	LEFT_ELBOW 		= 6
	LEFT_WRIST		= 7
	MID_HIP 		= 8
	RIGHT_HIP 		= 9
	RIGHT_KNEE 		= 10
	RIGHT_ANKLE 	= 11
	LEFT_HIP 		= 12
	LEFT_KNEE 		= 13
	LEFT_ANKLE 		= 14
	RIGHT_EYE 		= 15
	LEFT_EYE 		= 16
	RIGHT_EAR 		= 17
	LEFT_EAR 		= 18
	LEFT_BIG_TOE 	= 19
	LEFT_SMALL_TOE 	= 20
	LEFT_HEEL 		= 21
	RIGHT_BIG_TOE 	= 22
	RIGHT_SMALL_TOE = 23
	RIGHT_HEEL 		= 24
	BACKGROUND 		= 25


@unique
class trt_keypoints(Enum):
	NOSE 			= 0
	LEFT_EYE 		= 1
	RIGHT_EYE 		= 2
	LEFT_EAR 		= 3
	RIGHT_EAR 		= 4
	LEFT_SHOULDER 	= 5
	RIGHT_SHOULDER 	= 6
	LEFT_ELBOW 		= 7
	RIGHT_ELBOW 	= 8
	LEFT_WRIST		= 9
	RIGHT_WRIST 	= 10
	LEFT_HIP 		= 11
	RIGHT_HIP 		= 12
	LEFT_KNEE 		= 13
	RIGHT_KNEE 		= 14
	LEFT_ANKLE 		= 15
	RIGHT_ANKLE 	= 16
	NECK 			= 17

edges_for_skeleton_drawing = [("NOSE", "NECK"), ("NOSE", "LEFT_EYE"), ("NOSE", "RIGHT_EYE"), ("LEFT_EYE", "LEFT_EAR"), ("RIGHT_EYE", "RIGHT_EAR"),
	("NECK", "LEFT_SHOULDER"), ("NECK", "RIGHT_SHOULDER"),
	("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_SHOULDER", "LEFT_HIP"), ("LEFT_ELBOW", "LEFT_WRIST"),
	("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_SHOULDER", "RIGHT_HIP"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
	("LEFT_HIP", "RIGHT_HIP"), ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
	("LEFT_ANKLE", "LEFT_BIG_TOE"), ("LEFT_BIG_TOE", "LEFT_SMALL_TOE"), ("LEFT_ANKLE", "LEFT_HEEL"),
	("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"), ("RIGHT_ANKLE", "RIGHT_BIG_TOE"),
	("RIGHT_BIG_TOE", "RIGHT_SMALL_TOE"), ("RIGHT_ANKLE", "RIGHT_HEEL")
]

class SkeletonTracker:
	def __init__(self, module = 'openpose', track_body = True, track_hands = False, track_face = False, pipeline = None, choose_best_module = False, verbose = False):
		# If there are no available / installed module, raise an error
		if len(skeleton_tracking_modules) == 0:
			raise Exception("No module are available on this machine. Try installing openpose or cubemos for example.")

				# Allow the system to choose the best available module by
				# sorting by module's score
		if choose_best_module:
			sorted_modules = sorted(available_modules_importance, key=lambda mod: mod[1], reverse=True)
			self.module = sorted_modules[0][0]
		else:
			self.module = module.lower()

		self.pipeline = pipeline
		self.tracker = None
		self.track_body = track_body
		self.track_face = track_face
		self.track_hands = track_hands
		self._callbacks = {
			'initialized' : [],
			'process' : []
		}
		self._next_callback_id = 0

				# If the module is not available / installed, raise an
				# exception and propose other modules compatible
		if self.module.lower() not in map(lambda s: s.lower(), skeleton_tracking_modules): # Check if the desired module is available
			raise Exception(f"The module {module} is not available (yet). " + "\n\tPlease consider initializing SkeletonTracker with `module = [module_name]` using one of the following module names:\n" + '\n'.join(skeleton_tracking_modules))

		if verbose:
			print("Using module " + self.module)
		warnings = []
		if self.module == 'cubemos':
			if track_hands:
				warnings.append("The module Cubemos doesn't take into account the hand tracking.")
			if track_face:
				warnings.append("The module Cubemos doesn't take into account the face tracking.")

			if verbose:
				print("Loading model")
			self.tracker = skeletontracker(cloud_tracking_api_key="")

		if self.module == 'openpose':
			tracker_params = dict()
			tracker_params["model_folder"] = op.openpose_dir + "/models/"
			tracker_params["face"] = self.track_face
			tracker_params["hand"] = self.track_hands
			tracker_params["logging_level"] = 255
			# tracker_params["model_pose"] = "BODY_25"
			# tracker_params["model_pose"] = "COCO"
			tracker_params["model_pose"] = "MPI"
			# tracker_params["model_pose"] = "MPI_4_layers"
			tracker_params["net_resolution"] = "32x32"
			tracker_params["disable_multi_thread"] = True
			if verbose:
				print("Loading model")
			self.tracker = op.WrapperPython()
			# self.tracker = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)
			self.tracker.configure(tracker_params)
			self.tracker.start()

		if self.module == 'trt_pose':
			with open(trt_pose_data_dir + '/human_pose.json', 'r') as f:
				human_pose = json.load(f)

			topology = trt_pose.coco.coco_category_to_topology(human_pose)
			self._trt_object_parser = trt_pose.parse_objects.ParseObjects(topology)
			num_parts = len(human_pose['keypoints'])
			num_links = len(human_pose['skeleton'])

			# TODO let the user choose the resnet model
			# TODO2 : select cuda only if available
			original_model_weights = trt_pose_data_dir + '/resnet18_baseline_att_224x224_A_epoch_249.pth'
			optimized_model_weights = trt_pose_data_dir + '/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
			self._trt_width = 224
			self._trt_height = 224

			if not os.path.exists(optimized_model_weights):
				model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
				model.load_state_dict(torch.load(original_model_weights))
				# Size of the model
				self._trt_width = 224
				self._trt_height = 224
				# optimizing
				data = torch.zeros((1, 3, self._trt_height, self._trt_width)).cuda()
				model = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
				# Save the weights optimized for next time
				torch.save(model.state_dict(), optimized_model_weights)
			else:
				model = torch2trt.TRTModule()
				model.load_state_dict(torch.load(optimized_model_weights))

			self.tracker = model


		if len(warnings) > 0:
			print("Warning [SkeletonTracker] : " + '\n\t- '.join(warnings))

		self._last_frame_checked = None # Used to call multiple "track_skeletons" without re-computing the same frame twice
		self._last_results = {}

		if verbose:
			print("Model loaded")
		self._call_callback('initialized')

		self._tracking_thread = None

	def set_pipeline(self, pipeline):
		self.pipeline = pipeline

	def track_skeletons_async(self, image, use_cache = True, debug_time = False):
		if self._tracking_thread is None or not self._tracking_thread.is_alive():
			self._tracking_thread = threading.Thread(target = self.track_skeletons, kwargs = {'image':image, 'use_cache':use_cache, 'debug_time':debug_time}, daemon = False)
			self._tracking_thread.start()
		# if not self._tracking_thread.is_alive():
		# 	if self._tracking_thread.ident is None:
		# 		self._tracking_thread.start()
		# 	else:
		# 		self._tracking_thread.run(image = image, use_cache = use_cache, debug_time = debug_time)
		return self._last_results

	def track_skeletons(self, image, use_cache = True, debug_time = False):
		color_image = image
		if self.tracker is None:
			raise Exception("The skeleton tracker is not configured properly. See previous warnings.")

		if color_image is self._last_frame_checked:
			if use_cache and self._last_results is not None:
				results = self._last_results
		else:
			self._last_frame_checked = color_image

			start_time = time.time()
			results = None
			if self.module == 'cubemos':
				results = self._track_skeletons_cubemos(color_image)

			if self.module == 'openpose':
				results = self._track_skeletons_openpose(color_image)

			if self.module == 'trt_pose':
				results = self._track_skeletons_trt(color_image)


			self._last_results = results

			if debug_time:
				print("Inference time : ", int((time.time() - start_time) * 1000)/1000, "s")

		self._call_callback('process', skeleton_results = results)
		return results

	def _track_skeletons_cubemos(self, color_image):
		tracker_results = self.tracker.track_skeletons(color_image)
		results = dict()
		if self.track_body:
			results['body'] = []
			results['ids'] = []
			results['confidence'] = []
			for skeleton in tracker_results:
				results['body'].append(skeleton.joints)
				results['ids'].append(skeleton.id)
				results['confidence'].append(skeleton.confidences)
			results['body'] = np.array(results['body'])
			results['ids'] = np.array(results['ids'])
			results['confidence'] = np.array(results['confidence'])
		results['full'] = tracker_results # Add all the data, if needed
		return results

	def _track_skeletons_openpose(self, color_image):
		datum = op.Datum()
		datum.cvInputData = color_image
		self.tracker.emplaceAndPop(op.VectorDatum([datum]))
		results = dict()
		if self.track_body:
			results['body'] = []
			results['confidence'] = []
			results['ids'] = []
			if datum.poseKeypoints is not None:
				for i in range(len(datum.poseKeypoints)):
					results['body'].append(list(map(lambda res: [res[0], res[1]], datum.poseKeypoints[i]))) # Extract only the position
					results['confidence'].append(list(map(lambda res: res[2], datum.poseKeypoints[i])))		# Extract only the confidence
					results['ids'].append(datum.poseIds[i])
			results['body'] = np.array(results['body'])
			results['ids'] = np.array(results['ids'])
			results['confidence'] = np.array(results['confidence'])
		if self.track_hands:
			results['hands'] = datum.handKeypoints
		if self.track_face:
			results['face'] = datum.faceKeypoints

		results['full'] = datum  # Add all the data, if needed

		return results

	def _track_skeletons_trt(self, color_image):
		mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
		std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
		device = torch.device('cuda')
		image = cv2.resize(color_image.copy(), (self._trt_width, self._trt_height))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = PIL.Image.fromarray(image)
		image = transforms.functional.to_tensor(image).to(device)
		image.sub_(mean[:, None, None]).div_(std[:, None, None])
		image = image[None, ...]
		cmap, paf = self.tracker(image)
		cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
		counts, objects, peaks = self._trt_object_parser(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

		counts = counts[0]
		# print(int(counts), len(cmap[0]))
		#counts = 0 if counts == 0 else 1

		results = dict()
		if self.track_body:
			results['body'] = []
			results['confidence'] = []
			results['ids'] = []
			for i in range(counts):
				results['body'].append(self._trt_get_human_keypoints(objects, i, peaks, color_image.shape))
				results['confidence'].append(self._trt_get_confidence(cmap, peaks, i, objects))
				results['ids'].append(i)
			results['body'] = np.array(results['body'])
			results['ids'] = np.array(results['ids'])
			results['confidence'] = np.array(results['confidence'])
		results['full'] = peaks
		return results

	def _trt_get_human_keypoints(self, objects, index, peaks, image_dimensions):
		image_width = image_dimensions[1]
		image_height = image_dimensions[0]
		keypoints = []
		human = objects[0][index] # Index 0 because human is the first in topology
		for i in range(human.shape[0]): # For all keypoints
			j = int(human[i])
			if j >= 0:
				peak = peaks[0][i][j]
				keypoints.append([float(peak[1]) * image_width, float(peak[0]) * image_height])
			else:
				keypoints.append([-1, -1])
		return keypoints

	def _trt_get_confidence(self, cmap, peaks, index, objects):
		conf = []
		human = objects[0][index]
		for i in range(human.shape[0]): # i = num keypoint
			#conf.append(float(torch.max(cmap[index][i])))
			conf.append(float(torch.max(cmap[0][i]))) # It's not correct... only if 1 person in frame
		return conf


	def add_callback(self, action, callback, *args, **kwargs):
		action = action.lower()
		if action not in self._callbacks:
			raise Exception(f"The action `{action}` is not a callback method. Try one of these : " + "\n\t- ".join(list(self._callbacks.keys())))

		id = self._next_callback_id
		self._callbacks[action].append({'id' : id, 'function' : callback, 'args' : args, 'kwargs' : kwargs})
		self._next_callback_id += 1
		return id

	def _call_callback(self, action, **my_args):
		for callback in self._callbacks[action]:
			callback['function'](*callback['args'], **callback['kwargs'], **my_args) # Call all functions for this action

	def get_keypoint(self, name:str, part='body', person_id = None):
		keypoint_index = None
		keypoint_enum = None
		if self.module == 'cubemos':
			keypoint_enum = cubemos_keypoints
		elif self.module == 'openpose':
			keypoint_enum == openpose_keypoints
		elif self.module == 'trt':
			keypoint_enum = trt_keypoints
		try:
			keypoint_index = keypoint_enum[name.upper()].value
		except:
			raise Exception(f"No {name.upper()} keypoint defined in the module {self.module}. Try one of these instead:" + "\n" + "\n\t- ".join([k.name for k in keypoint_enum]))

		# If ID asked, return specific keypoint
		if person_id is not None:
			return self._last_results[part][person_id][keypoint_index]

		# Otherwise, return keypoint for all persons
		else:
			results = []
			for i, id in enumerate(self._last_results['ids']):
				results.append(self._last_results[part][i][keypoint_index])
			return results

	def get_keypoint3D(self, name:str, depth_map, part='body', person_id = None, pipeline = None):
		if pipeline is None:
			if self.pipeline is None:
				raise Exception("No pipeline defined, unable to get 3D coords")
			pipeline = self.pipeline
		import convert_realsense_to_real_world

		if person_id is not None: # Only one result
			return convert_realsense_to_real_world.get_3D_coord_from_realsense(self.get_keypoint(name, part, person_id), depth_map, pipeline)

		else:
			return [convert_realsense_to_real_world.get_3D_coord_from_realsense(coord, depth_map, pipeline) for coord in self.get_keypoint(name, part, None)]


	def draw_skeleton(self, color_image):
		skeletons = self.track_skeletons_async(color_image, use_cache=True, debug_time=False)
		if len(skeletons) > 0:
			# color_image = _colors_image.copy()

			keypoints_enum = None
			if self.module == 'openpose':
				keypoints_enum = openpose_keypoints
			if self.module == 'cubemos':
				keypoints_enum = cubemos_keypoints
			if self.module == 'trt_pose':
				keypoints_enum = trt_keypoints

			color_image = color_image.copy()

			possible_keypoints = [key.name for key in keypoints_enum]
			for i, id in enumerate(skeletons['ids']):
				for edge in edges_for_skeleton_drawing:
					if edge[0] not in possible_keypoints or edge[1] not in possible_keypoints:
						continue

					key0 = tuple(map(int, self.get_keypoint(edge[0], person_id=i)))
					key1 = tuple(map(int, self.get_keypoint(edge[1], person_id=i)))

					if key0 == (-1, -1) or key1 == (-1, -1):
						continue
					color_image = cv2.line(color_image, key0, key1, (0, 255, 255), 2)
				for key in possible_keypoints:
					color_image = cv2.circle(color_image, tuple(map(int, self.get_keypoint(key, person_id=i))), 4, (0, 255, 255), -1)
		return color_image


	def get_number_of_skeletons(self, color_image=None):
		if color_image is None:
			results = self._last_results
		else:
			results = self.track_skeletons(color_image, use_cache = True, debug_time = False)
		print(results)
		exit()
		return len(results['body'])

	def check_user_presence(self, color_image=None, required_confidence = None, min_number_of_joints_per_person = 0):
		if color_image is None:
			results = self._last_results
		else:
			results = self.track_skeletons(color_image, use_cache=True, debug_time=False)
		if len(results['ids']) == 0:
			return False

		max_confidence_found = -1.0
		for skeleton in results['confidence']:
			if min_number_of_joints_per_person > 0 and required_confidence is not None:
				number_of_confident_joints = len([c for c in skeleton if c >= required_confidence])
				if number_of_confident_joints < min_number_of_joints_per_person:
					continue

			max_confidence_found = max(max_confidence_found, max(skeleton))

		# if self._tracking_thread is not None and self._tracking_thread.is_alive():
		# 	self._tracking_thread.join()
		if required_confidence is not None:
			return max_confidence_found >= required_confidence
		else:
			return max_confidence_found > 0.0

	@property
	def is_3D_available(self):
		return self.pipeline is not None

