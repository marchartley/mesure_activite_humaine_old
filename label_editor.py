import json
import os
import threading
import tkinter as tk
import tkinter.filedialog
from collections import OrderedDict
from enum import Enum

import cv2
import ttkwidgets as ttk
from PIL import Image, ImageTk

import rosbag_to_videos as rtv

actions = ['walk', 'stand', 'sit', 'fall']



def get_filename_ext(filename):
	base = os.path.basename(filename)
	path = filename[:-(len(base) + 1)]
	name_splited = base.split(".")
	if len(name_splited) > 1:
		ext = name_splited[-1]
		base = base[:-(len(ext) + 1)]
	else:
		ext = None
	return path, base, ext


class AnnotationEditor(tk.Tk):
	def __init__(self):
		super().__init__()

		self.annotation_filename = "video_annotations.json"


		self.menu_bar = tk.Menu(self)
		self['menu'] = self.menu_bar

		file_menu = tk.Menu(self.menu_bar, tearoff = False)
		self.menu_bar.add_cascade(label = "File", menu = file_menu)
		file_menu.add_command(label = "Import video", command = self.import_video_command)

		self.btn_play = tk.Button(self, text = "Play")
		self.btn_label = tk.Button(self, text = "Label")

		self.action_buttons = []
		for i, action in enumerate(actions, 1):
			self.action_buttons.append(tk.Button(self, text = action + f" ({i})"))

		self.current_file_index = -1
		yDefilB = tk.Scrollbar(self, orient = 'vertical')
		yDefilB.grid(row = 0, column = 1)

		xDefilB = tk.Scrollbar(self, orient = 'horizontal')
		xDefilB.grid(row = 1, column = 0)
		self.list_of_filenames = []
		self.listbox_of_filenames = tk.Listbox(master = self, width = 40,
		                                       xscrollcommand = xDefilB.set,
		                                       yscrollcommand = yDefilB.set)
		self.listbox_of_filenames.bind("<<ListboxSelect>>", self.change_video_manually)
		xDefilB['command'] = self.listbox_of_filenames.xview
		yDefilB['command'] = self.listbox_of_filenames.yview

		self.video_canvas = VideoPlayer(self, width = 100)
		self.video_canvas.on_video_ended(self.wait_before_playing_next_video)

		self.video_slider = TimeLine(self,
		                             categories = {str(action): {"text": action} for action in actions},
		                             height = 100, extend = False, width = self.video_canvas.width,
		                             zoom_enabled = True, start = 0.0, finish = 0.0, unit = 's', resolution = 0.025,
		                             tick_resolution = 10.0)

		self.video_canvas.update_associated_slider(self.video_slider)

		self.listbox_of_filenames.grid(row = 0, column = 0, rowspan = 100)
		self.btn_play.grid(row = 0, column = 2)
		# self.btn_label.grid(row = 0, column = 3)
		for i, action_button in enumerate(self.action_buttons):
			action_button.grid(row = 0, column = 4 + i)
		self.video_canvas.grid(row = 1, column = 2, columnspan = 100)
		self.video_slider.grid(row = 2, column = 2, columnspan = 100)

		self.btn_play.config(command = self.video_canvas.switch_state)
		self.btn_label.config(command = lambda: self.video_slider.add_category('a', 'a'))

		self.protocol("WM_DELETE_WINDOW", self.handle_closing)

		self.set_video(None)
		self.video_has_been_changed_manually = False

		for i, action in enumerate(actions):
			self.bind(str(i+1), lambda evt, index=i: self.action_buttons[index].config(relief = tk.SUNKEN
			                              if self.action_buttons[index].cget('relief') == tk.RAISED else tk.RAISED))


	def set_video(self, filename, previous_video_id = -1):
		if previous_video_id != -1:
			self.add_annotation(self.annotation_filename, self.list_of_filenames[previous_video_id], self.video_canvas.annotations)
		if filename is not None:
			video_annotations = self.get_annotation_json(self.annotation_filename, filename)
		else:
			video_annotations = {}
		self.video_canvas.load_video_source(filename, VideoState.PAUSE, width = 600, height = 480, annotations = video_annotations)

	def change_video_manually(self, event):
		self.video_has_been_changed_manually = True
		previous_id = self.current_file_index
		self.current_file_index = self.listbox_of_filenames.curselection()[0]
		self.set_video(self.list_of_filenames[self.current_file_index], previous_video_id = previous_id)
		self.video_canvas.play()

	def handle_closing(self):
		if self.current_file_index != -1:
			self.add_annotation(self.annotation_filename, self.list_of_filenames[self.current_file_index], self.video_canvas.annotations)
		self.destroy()


	def import_video_command(self, preselected_files = None):
		if preselected_files is None:
			filenames = tk.filedialog.askopenfilenames(initialdir = os.getcwd(),
			                                           title = "Please select a video:",
			                                           filetypes = [# ('all', '*.*'),
				                                           ('special vids', '*as_color_video*.mp4'),
				                                           ('video', '*.mp4;*.avi')])
		else: # Debug mode
			filenames = preselected_files
		if len(filenames) > 0:
			for file in filenames:
				self.list_of_filenames.append(file)
				self.listbox_of_filenames.insert('end', file)
		self.listbox_of_filenames.selection_set(0)
		self.color_processed_video_filenames()

	def create_annotation_json(self, annotation_filename = "video_annotation.json", data = None):
		if data is None:
			data = {}
		json.dump(data, open(annotation_filename, "w+"), indent = 4)

	def add_annotation(self, annotation_filename = "video_annotation.json", video_filename = None, data = None):
		if data is None:
			raise Exception("You asked to create an annotation but you didn't provide any data.")
		if video_filename is None:
			raise Exception("You asked to create an annotation but you didn't provide the video filename.")

		annotations = self.get_annotation_json(annotation_filename)
		annotations[video_filename] = data
		self.create_annotation_json(annotation_filename, annotations)
		self.color_processed_video_filenames()

	def get_annotation_json(self, annotation_filename = "video_annotation.json", video_filename = None):
		if not os.path.exists(annotation_filename):
			self.create_annotation_json(annotation_filename)
		try:
			data = json.load(open(annotation_filename, "r+"))
		except json.JSONDecodeError: # If I removed manually all the content of the file
			data = {}
		if video_filename is not None:
			if video_filename in data:
				return data[video_filename]
			else:
				return {}
		else:
			return data

	def color_processed_video_filenames(self):
		data = self.get_annotation_json(self.annotation_filename)
		for i, filename in enumerate(self.list_of_filenames):
			if filename in data and len(data[filename]) > 0:
				self.listbox_of_filenames.itemconfigure(i, bg = "light green")
			else:
				self.listbox_of_filenames.itemconfigure(i, bg = "salmon")

	def wait_before_playing_next_video(self):
		self.video_has_been_changed_manually = False
		if self.current_file_index < len(self.list_of_filenames) - 1:
			timer = threading.Timer(3.0, self.play_next_video)
			timer.start()

	def play_next_video(self):
		if not self.video_has_been_changed_manually: # If the user didn't change video during the timer
			previous_index = self.current_file_index
			if self.current_file_index < len(self.list_of_filenames) - 1:
				self.current_file_index += 1
			self.listbox_of_filenames.select_clear(0, len(self.list_of_filenames))
			self.listbox_of_filenames.selection_set(self.current_file_index)
			self.set_video(self.list_of_filenames[self.current_file_index], previous_index)
			self.video_canvas.play()


class VideoState(Enum):
	PLAY = 0
	PAUSE = 1
	NO_VIDEO = 2


class VideoPlayer(tk.Canvas):
	def __init__(self, parent, filename = None, state = VideoState.PAUSE, associated_slider = None, width = None,
	             height = None):
		super().__init__(parent, bg = "black")
		self.filename = filename
		self.state = state
		self.current_frame = 0
		self.cap = None
		self.frame = None
		self.width = width if width is not None else 0
		self.height = height if height is not None else 0
		self.length = 0
		self.frame_as_image = None
		self.fps = -1
		self.frames_count = 0
		self.duration = -1

		self.associated_slider = associated_slider
		self.load_video_source(filename, state, resize_canvas = (width is not None or height is not None),
		                       width = width, height = height)
		self.refresh_rate = 1 / 30
		self.refresh_function_id = None
		self.refresh()

		self.mouse_click_event = None

		self.annotations = {}
		self.slider_updated_once = False

		self.callbacks = {}

		self.video_timestamp = None

		self.bind_all('<Key-Left>', self.backwards)
		self.bind_all('<Key-Right>', self.forward)

		self.current_action_markers = {}
		for i, action in enumerate(actions):
			self.current_action_markers[action] = -1
			self.bind_all(str(i+1), lambda evt, act=action: self.begin_or_end_action_marker(evt, act))

	def backwards(self, event, time_delta = 2.0):
		new_time = max(0, int(self.current_frame - time_delta * self.fps))
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_time)
	def forward(self, event, time_delta = 2.0):
		new_time = min(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), int(self.current_frame + time_delta * self.fps))
		self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_time)

	def load_video_source(self, new_filename = None, new_state = None, resize_canvas = True, width = 800, height = 600,
	                      annotations = None):
		if annotations is None:
			annotations = {}
		if new_filename:
			self.filename = new_filename
			if new_state is not None:
				self.state = new_state
			self.current_frame = 0

		if self.filename is not None:
			self.video_timestamp = rtv.get_timestamps_from_video(self.filename)
			self.cap = cv2.VideoCapture(self.filename)
			self.fps = self.cap.get(cv2.CAP_PROP_FPS)
			ret, frame = self.cap.read()
			self.frame = frame
			ratio = frame.shape[1] / frame.shape[0]  # w / h
			if width is None and height is None:
				self.width = frame.shape[1]
				self.height = frame.shape[0]
			else:
				if width is not None:
					self.width = width
				else:
					self.width = height * ratio

				if height is not None:
					self.height = height
				else:
					self.height = width / ratio

			self.frames_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
			self.duration = self.frames_count / self.fps

			if ret and resize_canvas:
				self.config(width = self.width, height = self.height)

			self.update_associated_slider(self.associated_slider)
			self.load_annotations(annotations)
			# if self.associated_slider is not None:
			# 	self.associated_slider.config(finish = self.duration, resolution = self.duration / self.width)
			return ret, frame
		else:
			return False, None

	def update_associated_slider(self, slider):
		self.associated_slider = slider
		self.associated_slider._canvas_ticks.bind("<B1-Motion>", self.slider_change_value)
		self.associated_slider._timeline.bind("<Button-1>", self._left_click_on_timeline)
		self.associated_slider._timeline.bind("<Button-3>", self._right_click_on_timeline)
		self.associated_slider._timeline.bind("<ButtonRelease-1>", self._left_click_released_on_timeline)
		if self.filename is not None:
			self.delete_marker(tk.ALL)
			self.associated_slider.config(finish = self.duration, resolution = self.duration / self.width, width = self.width)

	def _left_click_on_timeline(self, event):
		self.mouse_click_event = event

	# Override of the ttkwidget function
	def delete_marker(self, iid):
		if iid == tk.ALL:
			keys = list(self.associated_slider.markers.keys())
			for iid in keys:
				self.delete_marker(iid)
			return
		options = self.associated_slider._markers[iid]
		rectangle_id, text_id = options["rectangle_id"], options["text_id"]
		del self.associated_slider._canvas_markers[rectangle_id]
		if text_id in self.associated_slider._canvas_markers:  # Check for this value, it creates errors otherwise
			del self.associated_slider._canvas_markers[text_id]
		del self.associated_slider._markers[iid]
		self.associated_slider._timeline.delete(rectangle_id, text_id)

	def _right_click_on_timeline(self, event):
		x_time = self.associated_slider.get_position_time(event.x)
		category = [cat for cat in self.associated_slider._rows.keys() if
		            self.associated_slider._rows[cat][0] <= event.y <= self.associated_slider._rows[cat][1]]
		if len(category) == 0:
			return
		else:
			category = category[0]

		for i in self.associated_slider.markers:
			iid = self.associated_slider.markers[i]['iid']
			if category == self.associated_slider.markers[i]['category']:
				if self.associated_slider.markers[i]['start'] <= x_time <= self.associated_slider.markers[i]['finish']:
					self.delete_marker(iid)
					self.save_annotations()
					break

	def begin_or_end_action_marker(self, event, action_name):
		if self.current_action_markers[action_name] == -1: # Begin this action
			self.current_action_markers[action_name] = self.current_frame
		else: # Marker already started
			start_time = self.current_action_markers[action_name] / self.fps
			end_time = self.current_frame / self.fps
			self.associated_slider.create_marker(action_name, start_time, end_time)
			self.save_annotations()
			self.current_action_markers[action_name] = -1


	def _left_click_released_on_timeline(self, event):
		if self.associated_slider.current_iid is not None:
			return
		start_x = self.associated_slider._timeline.canvasx(self.mouse_click_event.x)
		end_x = self.associated_slider._timeline.canvasx(event.x)

		category = [cat for cat in self.associated_slider._rows.keys() if
		            self.associated_slider._rows[cat][0] <= event.y <= self.associated_slider._rows[cat][1]]
		if len(category) == 0:
			return
		else:
			category = category[0]
		mouse_move = [self.mouse_click_event.x - event.x, self.mouse_click_event.y - event.y]
		if abs(mouse_move[0]) > 0:  # A drag has been made
			self.associated_slider.create_marker(category,
			                                     self.associated_slider.get_position_time(start_x),
			                                     self.associated_slider.get_position_time(end_x))
		self.save_annotations()

	def save_annotations(self):
		self.annotations = {}
		timestamp_offset = 0
		if self.video_timestamp is not None:
			timestamp_offset = self.video_timestamp[0]

		for marker_id in self.associated_slider.markers:
			marker = self.associated_slider.markers[marker_id]
			if marker['category'] not in self.annotations:
				self.annotations[marker['category']] = []
			self.annotations[marker['category']].append({'from': marker['start'],
			                                             'to': marker['finish'],
			                                             'timestamp_start': marker['start'] + (timestamp_offset * 1000),
			                                             'timestamp_end': marker['finish'] + (timestamp_offset * 1000) })

	def load_annotations(self, annotations):
		self.annotations = annotations
		if self.associated_slider is not None:
			self.delete_marker(tk.ALL)
			for action in annotations:
				for annot in annotations[action]:
					self.associated_slider.create_marker(action, annot['from'], annot['to'])

	def move_time_slider_to_current_pos(self):
		if self.associated_slider is not None:
			self.associated_slider.set_time(self.current_frame / self.fps)

	def slider_change_value(self, event):
		self.associated_slider._time_marker_move(event)
		if self.cap:
			self.cap.set(cv2.CAP_PROP_POS_MSEC, self.associated_slider.time * 1000)

	def play(self):
		if self.filename is not None:
			self.state = VideoState.PLAY

	def pause(self):
		if self.filename is not None:
			self.state = VideoState.PAUSE

	def switch_state(self):
		if self.state == VideoState.PLAY:
			self.pause()
		elif self.state == VideoState.PAUSE:
			self.play()

	def update(self):
		if self.state == VideoState.PLAY:
			self.display_current_frame()
			self.move_time_slider_to_current_pos()
			super().update()

	def display_current_frame(self):
		ret, self.frame = self.cap.read()
		self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
		if not ret:
			self.state = VideoState.PAUSE
			for callback in self.callbacks['video_ends']:
				callback()
		else:
			scale = min(self.width / self.frame.shape[1], self.height / self.frame.shape[0])
			self.frame_as_image = ImageTk.PhotoImage(image = Image.fromarray(
				cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), (0, 0), fx = scale, fy = scale)))
			self.create_image(self.width / 2, self.height / 2, image = self.frame_as_image)

	def refresh(self):
		if self.refresh_function_id is not None:
			self.after_cancel(self.refresh_function_id)
		self.update()
		if self.associated_slider is not None:
			self.associated_slider.update()
		self.refresh_function_id = self.after(int(1 / self.fps * 1000), self.refresh)

	def on_video_ended(self, callback):
		if 'video_ends' not in self.callbacks:
			self.callbacks['video_ends'] = []
		self.callbacks['video_ends'].append(callback)


class TimeLine(ttk.TimeLine):
	def add_category(self, _id, name):
		self._categories['id'] = {'text': name}

		self.config({'categories': self._categories})
		self.update_display()

	def update_display(self):
		self._category_labels = OrderedDict()
		self.draw_categories()

		# Canvas widgets
		self._canvas_scroll = tk.Canvas(self, background = self._background, width = self._width, height = self._height)
		self._timeline = tk.Canvas(self._canvas_scroll, background = self._background, borderwidth = 0)
		self._timeline_id = self._canvas_scroll.create_window(0, 0, window = self._timeline, anchor = tk.NW)
		self._canvas_scroll.config(xscrollcommand = self._scrollbar_timeline.set,
		                           yscrollcommand = self._scrollbar_vertical.set)
		self._canvas_categories.config(yscrollcommand = self._scrollbar_vertical.set)

		self._setup_bindings()
		self.zoom_reset()
		self.draw_timeline()
		self.grid_widgets()
		self.update()


main = AnnotationEditor()
main.mainloop()
