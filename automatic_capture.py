import skeleton_tracker as tracker
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
import recording


class AutomaticCapture:
	def __init__(self, enable = True, tracker = None, pipeline = None, recording_dir = "", latency = 3): # Latency in sec
		self.AUTOMATIC_CAPTURE = enable
		self.tracker = None
		self.associate_tracker(tracker)
		self.pipeline = None
		self.associate_pipeline(pipeline)

		self._body_detected = False
		self.latency = latency
		self.last_seen_time = time.time()
		self._currently_capturing = False
		self.recording_dir = recording_dir

	def associate_tracker(self, tracker):
		self.tracker = tracker
		if tracker is not None:
			self.tracker.add_callback('process', self._check_if_body_found)

	def associate_pipeline(self, pipeline):
		self.pipeline = pipeline

	def set_automatic_capture(self, isAutoaticCapture):
		self.AUTOMATIC_CAPTURE = isAutoaticCapture

	def enable_automatic_capture(self):
		self.set_automatic_capture(True)

	def disable_automatic_capture(self):
		self.set_automatic_capture(False)

	def _check_if_body_found(self, skeleton_results):
		# Check for a body in an asynchronous way
		self._body_detected = self.tracker.check_user_presence(required_confidence = 0.5, min_number_of_joints_per_person = 3) # At least one keypoint found with > 0.5 confidence

	def update(self):
		if self.AUTOMATIC_CAPTURE:
			if self._body_detected:
				if not self.isCapturing():
					self.capture()
				self.last_seen_time = time.time()
			else:
				current_time = time.time()
				if current_time - self.last_seen_time >= self.latency:
					if self.isCapturing():
						self.stop_capture()
		return self._body_detected

	def capture(self):
		if self.pipeline is not None:
			if not self._currently_capturing:
				current_time = time.time()
				filename = self.recording_dir + "/"
				filename += time.strftime("%Y-%m-%d_%H_%M_%S_")
				filename += str(int((current_time - int(current_time)) * 1000)) + ".bag"
				self.pipeline = recording.start_recording(filename, self.pipeline)
				self._currently_capturing = True

	def stop_capture(self):
		if self.pipeline is not None:
			if self._currently_capturing:
				self.pipeline = recording.stop_recording(self.pipeline)
				self._currently_capturing = False

	def isCapturing(self):
		return self._currently_capturing


if __name__ == "__main__":
	# Example of usage: python automatic_capture.py --dir recordings --time-margin 5
	import argparse

	parser = argparse.ArgumentParser(description='Enable the RealSense camera and record a sequence if a person is detected')
	parser.add_argument('-d', '--dir',	dest='directory', metavar='directory', type=str,
					help='Directory where to save the sequences (current directory by default)',
					default = ".", required=False)
	parser.add_argument('-m', '--time-margin', dest='margin', help='Time (in second) elpased without seeing anyone before stopping the recording',
					type=float, default=10.0)

	args = parser.parse_args()
	directory = args.directory
	time_epsilon = args.margin

	pipeline = rs.pipeline()
	pipeline.start()
	skeletrack = tracker.SkeletonTracker(choose_best_module= True)

	auto_cap = AutomaticCapture(tracker = skeletrack, pipeline = pipeline, latency = time_epsilon, recording_dir = directory, enable = True)
	number_of_frames = 0
	start_time = time.time()
	fps = 0

	while True:
		try:
			frames = pipeline.wait_for_frames()
		except:
			break
		number_of_frames += 1
		if (time.time() - start_time) > 1 :
			fps = number_of_frames / (time.time() - start_time)
			number_of_frames = 0
			start_time = time.time()

		color_frame = np.asanyarray(frames.get_color_frame().get_data()).copy()
		depth_frame = np.asanyarray(frames.get_depth_frame().get_data()).copy()
		skeletons = skeletrack.track_skeletons_async(color_frame)
		auto_cap.update()

		color_frame = skeletrack.draw_skeleton(color_frame)
		cv2.putText(color_frame, "Status: " + ("recording" if auto_cap.isCapturing() else "waiting..."), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if auto_cap.isCapturing() else (0, 255, 0)), 2)
		cv2.putText(color_frame, "FPS: " + str(round(fps, 2)), (10, 50),
			cv2.FONT_HERSHEY_SIMPLEX, 1, ((0, 0, 255) if auto_cap.isCapturing() else (0, 255, 0)), 2)
		cv2.imshow('Realsense', cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q') or key == 27:
			break

	pipeline.stop()
	print(" --- End ---")
