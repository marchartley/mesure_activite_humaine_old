import pyrealsense2 as rs

precomputed_filters = {}


def apply_filters(depth_frame, filters = None):
	if filters is None:
		filters = []

	for rs_filter in filters:
		if rs_filter is not None:
			depth_frame = rs_filter.process(depth_frame)
	return depth_frame

def create_custom_filters(filter_name:str, filters: list) -> list:
	global precomputed_filters
	precomputed_filters[filter_name] = filters

def create_aruco_detection_filters() -> list:
	decimation = None # rs.decimation_filter(magnitude = 3.0)
	disparity2depth = rs.disparity_transform(True)
	temporal = rs.temporal_filter(smooth_alpha = 0.060, smooth_delta = 50, persistence_control = 8)
	spatial = rs.spatial_filter(smooth_alpha = 0.25, smooth_delta = 1.0, magnitude = 1.0, hole_fill = 0.0)
	depth2disparity = rs.disparity_transform(False)
	hole_filling = None # rs.hole_filling_filter(mode = 2)
	return [decimation, disparity2depth, spatial, temporal, depth2disparity, hole_filling]

def aruco_detection_filters(depth_frame):
	global precomputed_filters
	if 'aruco_detection' not in precomputed_filters:
		precomputed_filters['aruco_detection'] = create_aruco_detection_filters()

	return apply_filters(depth_frame, precomputed_filters['aruco_detection'])

def custom_filters(depth_frame, filter_name:str):
	global precomputed_filters
	if filter_name not in precomputed_filters:
		precomputed_filters[filter_name] = []

	return apply_filters(depth_frame, precomputed_filters['aruco_detection'])

