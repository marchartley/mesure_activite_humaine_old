import os

from moviepy.editor import concatenate_videoclips, ImageClip
import numpy as np
import pyrealsense2 as rs
import glob
import mutagen.mp4

import recording

import realsense_filters as filters

import InfluxDB_connection
import human_skeleton
import cv2

def get_filename_ext(filename):
	base = os.path.basename(filename)
	path = filename[:-(len(base) + 1)]
	name_splited = base.split(".")
	if len(name_splited) > 1:
		ext = name_splited[-1]
		base = base[:-(len(ext) + 1)]
	else:
		ext = None
	return (path, base, ext)

def save_video(frames: list, sensor_fps: int, target_fps: int, filename: str, metadata: dict = None):
	video_clip = concatenate_videoclips([ImageClip(m).set_duration(1 / sensor_fps) for m in frames])
	used_fps = target_fps
	if used_fps is None or used_fps <= 0:
		used_fps = sensor_fps
	video_clip.write_videofile(filename, fps = used_fps, logger = None)
	if metadata is not None:
		add_metadata_to_video(filename, metadata)
	return video_clip

def add_metadata_to_video(filename:str, data:dict):
	file = mutagen.mp4.MP4(filename)
	for key in data:
		file[key] = str(data[key])
	file.save()

def get_metadata_from_video(filename:str, key:str):
	file = mutagen.mp4.MP4(filename)
	if key in file:
		return file[key]
	else:
		return None

def get_timestamps_from_video(filename:str):
	file = mutagen.mp4.MP4(filename)
	start = float(file['start'])
	end = float(file['end'])
	return (start, end)

def create_video_from_rosbag(original_filename:str, color_video_destination:str, depth_video_destination:str = None, target_fps = -1, depth_colorization = 0):
	pipeline = recording.start_playback(original_filename, repeat_playback = False, realtime = False)
	frames_color = []
	frames_depth = []

	fps_color = pipeline.get_active_profile().get_stream(rs.stream.color).fps()
	fps_depth = pipeline.get_active_profile().get_stream(rs.stream.depth).fps()

	start_timestamp = None
	end_timestamp = None
	while True:
		try:
			frames = pipeline.wait_for_frames(timeout_ms = 500)
			if start_timestamp is None:
				start_timestamp = frames.timestamp
			end_timestamp = frames.timestamp
			frames.keep()
		except Exception as e:
			break

		if frames:
			color_frame = frames.get_color_frame().get_data()
			depth_frame = rs.colorizer(depth_colorization)\
				.colorize(filters.aruco_detection_filters(frames.get_depth_frame()))\
				.get_data()

			frames_color.append(np.array(color_frame))
			frames_depth.append(np.array(depth_frame))

	pipeline = recording.stop_playback(pipeline)

	if color_video_destination is not None:
		save_video(frames_color, fps_color, target_fps, color_video_destination, {'start': start_timestamp, 'end': end_timestamp})

	if depth_video_destination is not None:
		save_video(frames_depth, fps_depth, target_fps, depth_video_destination, {'start': start_timestamp, 'end': end_timestamp})

def create_video_for_all_persons(db_client:InfluxDB_connection.ListicClientDB, original_filename:str, color_video_destination:str, depth_video_destination:str = None, target_fps = -1, depth_colorization = 0):
	skeleton_db2D, skeleton_db3D = db_client.retrieve_skeleton_keypoints(original_filename)
	path_color, original_color_filename, ext_color = get_filename_ext(color_video_destination)
	path_depth, original_depth_filename, ext_depth = get_filename_ext(depth_video_destination)
	for skeleton_id in range(len(skeleton_db2D)):
		color_destination = os.path.join(path_color, original_color_filename + "_person_" + str(skeleton_id) + "." + ext_color)
		depth_destination = os.path.join(path_depth, original_depth_filename + "_person_" + str(skeleton_id) + "." + ext_depth)
		create_video_for_one_person(db_client, original_filename, skeleton_id, color_destination, depth_destination, target_fps, depth_colorization)


def create_video_for_one_person(db_client:InfluxDB_connection.ListicClientDB, original_filename:str, skeleton_id:int, color_video_destination:str, depth_video_destination:str = None, target_fps = -1, depth_colorization = 0):
	pipeline = recording.start_playback(original_filename, repeat_playback = False, realtime = False)
	frames_color = []
	frames_depth = []

	fps_color = pipeline.get_active_profile().get_stream(rs.stream.color).fps()
	fps_depth = pipeline.get_active_profile().get_stream(rs.stream.depth).fps()

	skeleton_db2D, skeleton_db3D = db_client.retrieve_skeleton_keypoints(original_filename)
	human = None

	body_has_been_seen = False

	start_timestamp = None
	end_timestamp = None
	current_timestamp = 0

	image_ratio = 4/3
	final_video_size = (800, 600)

	align = rs.align(rs.stream.color)

	while True:
		try:
			frames = pipeline.wait_for_frames(timeout_ms = 500)
			frames = align.process(frames)
			current_timestamp = frames.timestamp
			frames.keep()
		except Exception as e:
			break

		if frames:
			color_frame = np.array(frames.get_color_frame().get_data())
			depth_frame = np.array(rs.colorizer(depth_colorization).colorize(
				filters.aruco_detection_filters(frames.get_depth_frame())).get_data())

			try:
				human = human_skeleton.HumanSkeleton()
				human.from_json(data2D = skeleton_db2D[skeleton_id], data3D = skeleton_db3D[skeleton_id],
				                specific_timestamp = current_timestamp / 1000)  # InfluxDB mode
				bbox = human.get_bounding_box_2d(expand = 1.2)
				if bbox: # The person is here
					if not body_has_been_seen: # First frame where we see the person
						start_timestamp = frames.timestamp
					end_timestamp = frames.timestamp
					body_has_been_seen = True

					color_frame = cv2.line(color_frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[1])), (0, 255, 255), 2)
					color_frame = cv2.line(color_frame, tuple(map(int, bbox[1])), tuple(map(int, bbox[3])), (0, 255, 255), 2)
					color_frame = cv2.line(color_frame, tuple(map(int, bbox[3])), tuple(map(int, bbox[2])), (0, 255, 255), 2)
					color_frame = cv2.line(color_frame, tuple(map(int, bbox[2])), tuple(map(int, bbox[0])), (0, 255, 255), 2)

					depth_frame = cv2.line(depth_frame, tuple(map(int, bbox[0])), tuple(map(int, bbox[1])), (0, 255, 255), 2)
					depth_frame = cv2.line(depth_frame, tuple(map(int, bbox[1])), tuple(map(int, bbox[3])), (0, 255, 255), 2)
					depth_frame = cv2.line(depth_frame, tuple(map(int, bbox[3])), tuple(map(int, bbox[2])), (0, 255, 255), 2)
					depth_frame = cv2.line(depth_frame, tuple(map(int, bbox[2])), tuple(map(int, bbox[0])), (0, 255, 255), 2)

					width, height = bbox[3][0] - bbox[0][0], bbox[3][1] - bbox[0][1]
					center = np.array([bbox[0][0] + width/2, bbox[0][1] + height/2])

					crop_height = height
					crop_width = crop_height * image_ratio
					if center[0] - crop_width/2 < 0:
						crop_min_x = 0
						crop_max_x = crop_width
					elif center[0] - crop_width/2 > color_frame.shape[1]:
						crop_max_x = color_frame.shape[1]
						crop_min_x = crop_max_x - crop_width
					else:
						crop_min_x = center[0] - crop_width/2
						crop_max_x = center[0] + crop_width/2

					if center[1] - crop_height / 2 < 0:
						crop_min_y = 0
						crop_max_y = crop_height
					elif center[1] - crop_height / 2 > color_frame.shape[0]:
						crop_max_y = color_frame.shape[0]
						crop_min_y = crop_max_y - crop_height
					else:
						crop_min_y = center[1] - crop_height / 2
						crop_max_y = center[1] + crop_height / 2

					crop_min_x = int(crop_min_x)
					crop_max_x = int(crop_max_x)
					crop_min_y = int(crop_min_y)
					crop_max_y = int(crop_max_y)

					cropped_color_frame = color_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
					cropped_depth_frame = depth_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
					cropped_color_frame = cv2.resize(cropped_color_frame, final_video_size)
					cropped_depth_frame = cv2.resize(cropped_depth_frame, final_video_size)
					frames_color.append(cropped_color_frame)
					frames_depth.append(cropped_depth_frame)

				else: # The person is not in the frame
					is_last_frame_with_body = not InfluxDB_connection.ListicClientDB.check_if_body_will_reapear(skeleton_db2D[skeleton_id], current_timestamp/1000)
					if is_last_frame_with_body:
						break # Stop the recording if the body won't reappear

					if body_has_been_seen: # Person is temporarly unvisible, replace with the last frame
						frames_color.append(frames_color[-1])
						frames_depth.append(frames_depth[-1])
						# frames_color.append(np.zeros((final_video_size[0], final_video_size[1], 3), dtype=int))
						# frames_depth.append(np.zeros((final_video_size[0], final_video_size[1], 3), dtype=int))
					else: # Person didn't appeard yet
						pass

			except Exception as e:
				pass

	pipeline = recording.stop_playback(pipeline)

	if color_video_destination is not None:
		save_video(frames_color, fps_color, target_fps, color_video_destination, {'start': start_timestamp, 'end': end_timestamp})

	if depth_video_destination is not None:
		save_video(frames_depth, fps_depth, target_fps, depth_video_destination, {'start': start_timestamp, 'end': end_timestamp})



if __name__ == "__main__":
	# Example of usage: python rosbag_to_videos.py --dir recordings
	import argparse

	parser = argparse.ArgumentParser(description='Transform a ROS-bag sequence into a RGB mp4 and DEPTH mp4')
	files_group = parser.add_mutually_exclusive_group(required = True)
	files_group.add_argument('-f', '--filename',	dest='filename', metavar='filename', type=str,
							help='Transform a specific sequence .bag', required=False, default=None)
	files_group.add_argument('-d', '--dir',	dest='directory', metavar='directory', type=str,
							help='Transform the whole directory', required=False, default=None)
	parser.add_argument('--output-color', '-oc', dest = 'color_output_dir',
	                    help = 'Directory to store the color version of the video',
	                    type = str, default = None)
	parser.add_argument('--output-depth', '-od', dest = 'depth_output_dir',
	                    help = 'Directory to store the depth version of the video',
	                    type = str, default = None)
	parser.add_argument('--target-fps', dest = 'target_fps',
	                    help = 'Amount of frames per second in the resulting video file',
	                    type = int, default = 10)
	parser.add_argument('--no-color', dest = 'no_color', help = 'Do not create the color version of the video',
	                    action = 'store_true', default = False)
	parser.add_argument('--no-depth', dest = 'no_depth', help = 'Do not create the depth version of the video',
	                    action = 'store_true', default = False)
	parser.add_argument('--color-suffix', '-cs', dest = 'color_suffix',
	                    help = 'Add a suffix at the end of the file name for the color version of the video',
	                    type = str, default = '_as_color_video')
	parser.add_argument('--depth-suffix', '-ds', dest = 'depth_suffix',
	                    help = 'Add a suffix at the end of the file name for the depth version of the video',
	                    type = str, default = '_as_depth_video')
	args = parser.parse_args()
	filename = args.filename
	directory = args.directory
	target_fps = args.target_fps
	no_depth = args.no_depth
	no_color = args.no_color
	depth_suffix = args.depth_suffix
	color_suffix = args.color_suffix
	depth_output_dir = args.depth_output_dir
	if depth_output_dir is None:
		if directory is not None:
			depth_output_dir = directory
		else:
			depth_output_dir = get_filename_ext(filename)[0] # Same directory as the file
	color_output_dir = args.color_output_dir
	if color_output_dir is None:
		if directory is not None:
			color_output_dir = directory
		else:
			color_output_dir = get_filename_ext(filename)[0] # Same directory as the file

	db_client = InfluxDB_connection.ListicClientDB(server_side = False)
	all_files = []
	if directory is None:
		all_files = [filename]
	else:
		all_files = glob.glob(directory + "/*.bag")

	print(f"Converting {len(all_files)} files:")
	for file_name in all_files:
		print(f" - {file_name}")

	for i, file_name in enumerate(all_files):
		print(f"{round((i / len(all_files))*100)}%")

		path, original_filename, ext = get_filename_ext(file_name)
		if no_color:
			color_filename = None
		else:
			color_filename = os.path.join(color_output_dir, original_filename + color_suffix + ".mp4")

		if no_depth:
			depth_filename = None
		else:
			depth_filename = os.path.join(depth_output_dir, original_filename + depth_suffix + ".mp4")

		# Transform the rosbag into a video, without looking for skeletons
		# create_video_from_rosbag(file_name, color_filename, depth_filename, target_fps = target_fps, depth_colorization = 0)

		# Transform into a video and crop to follow one specific person
		# create_video_for_one_person(db_client, file_name, 0, color_filename, depth_filename, target_fps = target_fps, depth_colorization = 0)

		# Transform into multiple videos, each one following one different person
		create_video_for_all_persons(db_client, file_name, color_filename, depth_filename, target_fps = target_fps, depth_colorization = 0)
	print("100%")
