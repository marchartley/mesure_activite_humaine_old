import cv2
import skeleton_tracker as st
import pyrealsense2 as rs
import numpy as np
import colorsys
import time
import json

import influxdb as db

import matplotlib.pyplot as plt

from convert_realsense_to_real_world import *

import human_skeleton

import InfluxDB_connection
import datetime
import glob
import math

def save_data(bag_filename, client = None, skeletrack = None, device = 'realsense1'):
	if client is None:
		client = InfluxDB_connection.ListicClientDB(server_side = False) # Connect locally if no InfluxDB client is given
		print("No specific InfluxDB client is given, continuing with local client")


	pipeline = rs.pipeline()

	if skeletrack is None:
		skeletrack = st.SkeletonTracker(choose_best_module=True, pipeline = pipeline)
	else:
		skeletrack.set_pipeline(pipeline)
	pipeline = start_playback(bag_filename, pipeline = pipeline, realtime = False)
	joint_confidence = 0.4


	start_time = time.time()
	human = None

	print("Starting...")

	points = []

	while True:
		try:
			frames = pipeline.wait_for_frames(timeout_ms = 500)
		except:
			break
		color_frame = frames.get_color_frame()
		depth_frame = frames.get_depth_frame()

		color_frame = np.asanyarray(color_frame.get_data()).copy()

		skeletons = skeletrack.track_skeletons(color_frame, debug_time = False)

		if skeletrack.check_user_presence(color_frame):
			for i in range(skeletrack.get_number_of_skeletons()):
				human = human_skeleton.HumanSkeleton()
				human.from_tracker(skeletrack, i, depth_frame)

				coord2D, coord3D = human.to_json()

				client.insert_skeleton_keypoints(coord2D = coord2D, coord3D = coord3D, skeleton_id = i, timestamp = frames.timestamp, device = device, send_immediatly = False)
	client.send_to_db()
	print("Parsing video in ", round(time.time() - start_time, 2), "s")

	pipeline = stop_playback(pipeline)

def retrieve_data(bag_filename, client = None, verbose = False):
	if client is None:
		client = InfluxDB_connection.ListicClientDB(server_side = False) # Connect locally if no InfluxDB client is given
		print("No specific InfluxDB client is given, continuing with local client")

	start = time.time()
	res2D, res3D = client.retrieve_skeleton_keypoints(bag_filename)
	end = time.time()
	if verbose:
		number_of_keypoints = 0
		start_time = math.inf
		end_time = -math.inf
		for skeleton in res2D:
			for key in skeleton:
				number_of_keypoints += len(skeleton[key]) # Number of frame per keypoint
				start_time = min(start_time, skeleton[key][0]['time_ms'])
				end_time = max(end_time, skeleton[key][-1]['time_ms'])
			# number_of_keypoints *= (3 + 2) # 3 coord for 3D + 2 coord for 2D
		print(f"Fetching {number_of_keypoints} keypoints : {round(end - start, 2)}s [duration = {(end_time - start_time)}s]")



if __name__ == "__main__":
	# Example of usage: python save_skeleton_to_server.py --dir recordings --local
	# or    python save_skeleton_to_server.py --dir recordings --remote --adress 192.168.143.29
	import argparse

	parser = argparse.ArgumentParser(description='Open all ros-bag files from a folder, run a skeleton detector on them and save the keypoints on the InfluxDB server')
	file_group = parser.add_mutually_exclusive_group(required = True)
	file_group.add_argument('-d', '--dir',	dest='directory', metavar='directory', type=str,
					help='Directory where the sequences are stored',
					default = "recordings")
	file_group.add_argument('-f', '--filename',	dest='filename', metavar='filename', type=str,
					help='File on which you want to run the skeleton detection and send keypoints')
	host_group = parser.add_mutually_exclusive_group(required = True)
	host_group.add_argument('--local', dest='local', help='Use the InfluxDB installed on this machine', action='store_true')
	host_group.add_argument('--remote', dest='remote', help='Use the InfluxDB installed on a remote machine', action='store_true')
	parser.add_argument('--adress', dest='adress', help='If using --remote, define the server adress', type=str)
	parser.add_argument('--port', dest='port', help='Define the port used by InfluxDB (default: 8086)', type=str, default = 8086)
	parser.add_argument('-l', '--login', dest='login', help='Username for InfluxDB', type=str)
	parser.add_argument('-p', '--password', dest='password', help='Password for InfluxDB', type=str)
	parser.add_argument('-db', '--database', dest='database', help='Database to connect to in InfluxDB', type=str)

	args = parser.parse_args()
	recordings_folder = args.directory
	local = args.local
	remote = args.remote
	adress = args.adress
	port = args.port
	login = args.login
	password = args.password
	database = args.database
	filename = args.filename

	if adress is None:
		if local:
			adress = 'localhost'
		elif remote:
			parser.error('--adress is required if --remote is chosen')
	if login is not None and password is None:
		parser.error('--password is required with --login')

	db_client = InfluxDB_connection.ListicClientDB(server_side = remote, host = adress, port = port, login = login, pwd = password, database = database)

	if filename is not None:
		save_data(filename, client = db_client)
		retrieve_data(filename, client = db_client)
		plt.show()
	else:
		files_list = glob.glob(recordings_folder + "/*.bag")
		print(f"{len(files_list)} files found")
		for file in files_list:
			print(" -", file)

		skeletrack = st.SkeletonTracker(choose_best_module = True)
		for i, file in enumerate(files_list):
			print(f"{round((i/len(files_list))*100)}%")
			save_data(file, client = db_client, skeletrack = skeletrack)
			# retrieve_data(file, client = db_client)
		print("100%")
	# for i, file in enumerate(glob.glob(recordings_folder + "/*.bag"), 1):
	# 	plt.figure(i)
	# 	retrieve_data(file, client = db_client)
	# plt.show()




