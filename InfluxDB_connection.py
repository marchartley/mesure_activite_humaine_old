import influxdb as db
import numpy as np
import recording
import datetime
import math


class ListicClientDB():
	def __init__(self, server_side = None, host = None, port = None, login = None, pwd = None, database = None):
		self.stored_points = []
		self.client = None

		if server_side:
			if host is None:
				host = '192.168.143.29'
			if port is None:
				port = 8086
			if login is None:
				login = 'listic'
			if pwd is None:
				pwd = 'groupefiad'
			if database is None:
				database = 'iot'
		else:
			if host is None:
				host = 'localhost'
			if port is None:
				port = 8086
			if login is None:
				login = 'admin'
			if pwd is None:
				pwd = 'admin'
			if database is None:
				database = 'iot'

		self.client = db.InfluxDBClient(host, port, login, pwd)
		self.client.switch_database(database)
		try:
			self.client.ping()
		except Exception as e:
			print("The DB is unreachable...")
			raise e

	def insert_skeleton_keypoints(self, coord2D = None, coord3D = None, timestamp = None, skeleton_id = 0,
									device = None, ignore_None_values = True, send_immediatly = True,
									timestamp_unit = 'ms'):
		if self.client is None:
			raise Exception(f"Client of {self.__class__.__name__} wasn't configure correctly")

		if coord2D is not None:
			for i, key in enumerate(coord2D):
				if ignore_None_values and np.all(coord2D[key]) is not None:
					self.stored_points.append({
						'measurement': 'skeleton_detection_2d',
						'time': round(timestamp),
						'tags': {
							'device': device,
							'skeleton_id': skeleton_id,
							'keypoint': key
						},
						'fields': {
							'x': coord2D[key][0],
							'y': coord2D[key][1]
						}
					})
		if coord3D is not None:
			for i, key in enumerate(coord3D):
				if ignore_None_values and np.all(coord3D[key]) is not None:
					self.stored_points.append({
						'measurement': 'skeleton_detection_3d',
						'time': round(timestamp),
						'tags': {
							'device': device,
							'skeleton_id': skeleton_id,
							'keypoint': key
						},
						'fields': {
							'x': coord3D[key][0],
							'y': coord3D[key][1],
							'z': coord3D[key][2]
						}
					})

		if send_immediatly:
			self.send_to_db()

	def send_to_db(self, timestamp_unit = 'ms'):
		self.client.write_points(self.stored_points, time_precision = timestamp_unit)
		self.stored_points = []

	def retrieve_skeleton_keypoints(self, sequence_filename = None, start_time = None, end_time = None):
		if start_time is None and end_time is None:
			if sequence_filename is None:
				raise Exception(
					"Cannot retrieve skeleton keypoints because no filename has been given, nor start/end time")
			else:
				pipeline = recording.start_playback(sequence_filename, realtime = False)
				start_time = pipeline.wait_for_frames().timestamp
				duration = pipeline.get_active_profile().get_device().as_playback().get_duration()
				pipeline = recording.stop_playback(pipeline)
				duration = (duration / datetime.timedelta(microseconds = 1)) / 1e3
				end_time = start_time + duration

		start_time = math.floor(start_time)
		end_time = math.ceil(end_time)

		query2D = 'SELECT "x", "y" FROM "autogen"."skeleton_detection_2d" WHERE ' + str(
			start_time) + 'ms <= time and time <= ' + str(end_time) + 'ms  GROUP BY "skeleton_id", "keypoint"'
		query3D = 'SELECT "x", "y", "z" FROM "autogen"."skeleton_detection_3d" WHERE ' + str(
			start_time) + 'ms <= time and time <= ' + str(end_time) + 'ms  GROUP BY "skeleton_id", "keypoint"'

		results2D = self.client.query(query2D)
		points2D = {}
		for key, point in results2D.items():
			skeleton_id = int(key[1]['skeleton_id'])
			if skeleton_id not in points2D:
				points2D[skeleton_id] = {}
			keypoint = key[1]['keypoint']
			points2D[skeleton_id][keypoint] = np.array(list(map(  # Add the time as timestamp instead of str
				lambda coord:
				dict(coord,
				     **{'time_ms': datetime.datetime.fromisoformat(coord['time'].replace('Z', '+00:00')).timestamp()}),
				point)))

		results3D = self.client.query(query3D)
		points3D = {}
		for key, point in results3D.items():
			skeleton_id = int(key[1]['skeleton_id'])
			if skeleton_id not in points3D:
				points3D[skeleton_id] = {}
			keypoint = key[1]['keypoint']
			points3D[skeleton_id][keypoint] = np.array(list(map(  # Add the time as timestamp instead of str
				lambda coord:
				dict(coord,
				     **{'time_ms': datetime.datetime.fromisoformat(coord['time'].replace('Z', '+00:00')).timestamp()}),
				point)))

		return points2D, points3D

	@staticmethod
	def check_if_body_will_reapear(skeleton_data:dict, current_timestamp:float):
		max_timestamp = 0
		for keypoint in skeleton_data:
			max_timestamp = max(max_timestamp, max([data['time_ms'] for data in skeleton_data[keypoint]]))

		return max_timestamp > current_timestamp
