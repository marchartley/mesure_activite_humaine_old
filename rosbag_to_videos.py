import moviepy
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips, ImageClip
import cv2
import numpy as np
import pyrealsense2 as rs
import glob

import recording

import realsense_filters as filters

def get_filename_ext(filename):
	name_splited = filename.split(".")
	filename = ".".join(name_splited[:-1])
	if len(name_splited) > 1:
		ext = name_splited[-1]
	else:
		ext = None
	return (filename, ext)


if __name__ == "__main__":
	# Example of usage: python rosbag_to_videos.py --dir recordings
	import argparse

	parser = argparse.ArgumentParser(description='Transform a ROS-bag sequence into a RGB mp4 and DEPTH mp4')
	files_group = parser.add_mutually_exclusive_group(required = True)
	files_group.add_argument('-f', '--filename',	dest='filename', metavar='filename', type=str,
							help='Transform a specific sequence .bag', required=False, default=None)
	files_group.add_argument('-d', '--dir',	dest='directory', metavar='directory', type=str,
							help='Transform the whole directory', required=False, default=None)
	parser.add_argument('--target-fps', dest = 'target_fps',
	                    help = 'Amount of frames per second in the resulting video file',
	                    type = int, default = 10)
	args = parser.parse_args()
	filename = args.filename
	directory = args.directory
	target_fps = args.target_fps

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
		pipeline = recording.start_playback(file_name, repeat_playback = True, realtime = False)
		frames_color = []
		frames_depth = []

		fps_color = pipeline.get_active_profile().get_stream(rs.stream.color).fps()
		fps_depth = pipeline.get_active_profile().get_stream(rs.stream.depth).fps()

		frame_count = 0
		while True:
			try:
				frames = pipeline.wait_for_frames(timeout_ms = 500)
				frames.keep()
			except Exception as e:
				print(e)
				break

			if frames:
				color_frame = frames.get_color_frame().get_data()
				# depth_frame = rs.colorizer(2).colorize(frames.get_depth_frame()).get_data()
				depth_frame = rs.colorizer().colorize(filters.custom_filter(frames.get_depth_frame())).get_data()
				frame_count += 1
				# cv2.imshow('depth', cv2.cvtColor(np.array(depth_frame), cv2.COLOR_BGR2RGB))
				frames_color.append(np.array(color_frame))
				frames_depth.append(np.array(depth_frame))
				cv2.imshow("depth", cv2.cvtColor(np.array(depth_frame), cv2.COLOR_BGR2RGB))

			if cv2.waitKey(1) & 0xFF == ord('q'):
				pipeline = recording.stop_playback(pipeline)
				break

		pipeline = recording.stop_playback(pipeline)

		file_name, _ = get_filename_ext(file_name)
		color_filename = file_name + "_color.mp4"
		depth_filename = file_name + "_depth.mp4"

		color_clip = concatenate_videoclips([ImageClip(m).set_duration(1/fps_color) for m in frames_color])
		color_clip.write_videofile(color_filename, fps=target_fps, logger=None)

		depth_clip = concatenate_videoclips([ImageClip(m).set_duration(1/fps_depth) for m in frames_depth])
		depth_clip.write_videofile(depth_filename, fps=target_fps, logger=None)

	print("100%")
