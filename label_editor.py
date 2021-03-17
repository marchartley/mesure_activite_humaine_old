import tkinter as tk 
import ttkwidgets as ttk
import cv2
from enum import Enum
from PIL import Image, ImageTk

from collections import OrderedDict
from ttkwidgets import AutoHideScrollbar

actions = ['walk', 'stand', 'sit', 'fall']

def create_editor():
	win = tk.Tk()


	btn_play = tk.Button(win, text="Play")
	btn_label = tk.Button(win, text="Label")

	video_canvas = VideoPlayer(win, "test_file_color.mp4", width = 400)

	video_slider = TimeLine(win,
								categories={str(action): {"text": action} for action in actions},
								height=100, extend=False, width = video_canvas.width,
								zoom_enabled=True, start = 0.0, finish = 0.0, unit='s', resolution = 0.025, tick_resolution=10.0)#, from_=0, to=100, orient=tk.HORIZONTAL, length = video_canvas.width)
	

	video_canvas.update_associated_slider(video_slider)
	# video_slider.create_marker("walk", 0, 20)
	video_slider.create_marker("sit", 4, 20)



	btn_play.grid(row = 0, column = 0)
	btn_label.grid(row = 0, column = 1)
	video_canvas.grid(row = 1, column = 0, columnspan = 100)
	video_slider.grid(row = 2, column = 0, columnspan = 100)

	btn_play.config(command=video_canvas.switch_state)
	btn_label.config(command=lambda: video_slider.add_category('a', 'a'))
	# video_slider.add_category('a', 'a')
	return win

# label_id = 0
# def add_label(slider):
# 	global actions, label_id
# 	actions.append("Test" + str(label_id))
# 	label_id += 1
# 	slider.config({"categories":{str(action): {"text": action} for action in actions}})
# 	# Category Labels
# 	slider._category_labels = OrderedDict()
# 	slider.draw_categories()

# 	# Canvas widgets
# 	slider._canvas_scroll = tk.Canvas(slider, background=slider._background, width=slider._width, height=slider._height)
# 	slider._timeline = tk.Canvas(slider._canvas_scroll, background=slider._background, borderwidth=0)
# 	slider._timeline_id = slider._canvas_scroll.create_window(0, 0, window=slider._timeline, anchor=tk.NW)
# 	# if slider._autohidescrollbars:
# 	# 	slider._scrollbar_timeline = AutoHideScrollbar(slider, command=slider._set_scroll, orient=tk.HORIZONTAL)
# 	# 	slider._scrollbar_vertical = AutoHideScrollbar(slider, command=slider._set_scroll_v, orient=tk.VERTICAL)
# 	# else:
# 	# 	slider._scrollbar_timeline = ttk.Scrollbar(slider, command=slider._set_scroll, orient=tk.HORIZONTAL)
# 	# 	slider._scrollbar_vertical = ttk.Scrollbar(slider, command=slider._set_scroll_v, orient=tk.VERTICAL)
# 	slider._canvas_scroll.config(xscrollcommand=slider._scrollbar_timeline.set,
# 							   yscrollcommand=slider._scrollbar_vertical.set)
# 	slider._canvas_categories.config(yscrollcommand=slider._scrollbar_vertical.set)

# 	slider._setup_bindings()
# 	slider.zoom_reset()
# 	slider.draw_timeline()
# 	slider.grid_widgets()

class VideoState(Enum):
	PLAY = 0
	PAUSE = 1
	NO_VIDEO = 2

class VideoPlayer(tk.Canvas):
	def __init__(self, parent, filename = None, state = VideoState.PAUSE, associated_slider = None, width = None, height = None):
		super().__init__(parent, bg="black")
		self.filename = filename
		self.state = state
		self.current_frame = 0
		self.cap = None
		self.frame = None
		self.width = width if width is not None else 0
		self.height = height if height is not None else 0
		self.length = 0
		self.associated_slider = associated_slider
		self.load_video_source(filename, state, resize_canvas = (width is not None or height is not None), width = width, height = height)

		self.refresh_rate = 1/30
		self.refresh_function_id = None
		self.refresh()

	def load_video_source(self, new_filename = None, new_state = None, resize_canvas = True, width = 800, height = 600):
		if new_filename:
			self.filename = new_filename
			if new_state is not None:
				self.state = new_state
			self.current_frame = 0

		if self.filename is not None:
			self.cap = cv2.VideoCapture(self.filename)
			self.fps = self.cap.get(cv2.CAP_PROP_FPS)
			ret, frame = self.cap.read()
			self.frame = frame
			ratio = frame.shape[1]/frame.shape[0] # w / h
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

			if self.associated_slider is not None:
				self.associated_slider.config(finish = self.duration, resolution =  self.duration / self.width)
			return ret, frame
		else:
			return False, None

	def update_associated_slider(self, slider):
		self.associated_slider = slider
		self.associated_slider._canvas_ticks.bind("<B1-Motion>", self.slider_change_value)
		self.associated_slider.config(finish = self.duration, resolution = self.duration / self.width)

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
		else:
			scale = min(self.width / self.frame.shape[1], self.height / self.frame.shape[0])
			self.frame_as_image =  ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), (0, 0), fx = scale, fy = scale)))
			self.create_image(self.width/2, self.height/2, image = self.frame_as_image)


	def refresh(self):
		if self.refresh_function_id is not None:
			self.after_cancel(self.refresh_function_id)
		self.update()
		if self.associated_slider is not None:
			self.associated_slider.update()
		self.refresh_function_id = self.after(int(1/self.fps * 1000), self.refresh)


class TimeLine(ttk.TimeLine):
	def add_category(self, id, name):
		self._categories['id'] = {'text' : name}

		self.config({'categories': self._categories})
		self.update_display()

	def update_display(self):

		self._category_labels = OrderedDict()
		self.draw_categories()

		# Canvas widgets
		self._canvas_scroll = tk.Canvas(self, background=self._background, width=self._width, height=self._height)
		self._timeline = tk.Canvas(self._canvas_scroll, background=self._background, borderwidth=0)
		self._timeline_id = self._canvas_scroll.create_window(0, 0, window=self._timeline, anchor=tk.NW)
		# if self._autohidescrollbars:
		# 	self._scrollbar_timeline = AutoHideScrollbar(self, command=self._set_scroll, orient=tk.HORIZONTAL)
		# 	self._scrollbar_vertical = AutoHideScrollbar(self, command=self._set_scroll_v, orient=tk.VERTICAL)
		# else:
		# 	self._scrollbar_timeline = ttk.Scrollbar(self, command=self._set_scroll, orient=tk.HORIZONTAL)
		# 	self._scrollbar_vertical = ttk.Scrollbar(self, command=self._set_scroll_v, orient=tk.VERTICAL)
		self._canvas_scroll.config(xscrollcommand=self._scrollbar_timeline.set,
								   yscrollcommand=self._scrollbar_vertical.set)
		self._canvas_categories.config(yscrollcommand=self._scrollbar_vertical.set)

		self._setup_bindings()
		self.zoom_reset()
		self.draw_timeline()
		self.grid_widgets()
		self.update()




main = create_editor()
main.mainloop()