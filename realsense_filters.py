import pyrealsense2 as rs

previously_filtered_frames = []

def apply_filters(depth_frame, filters = None):
	if filters is None:
		filters = []

	for rs_filter in filters:
		if rs_filter is not None:
			depth_frame = rs_filter.process(depth_frame)
	return depth_frame


def custom_filter(depth_frame, number_of_filters_for_temporal_filter = 3):
	global previously_filtered_frames
	if len(previously_filtered_frames) > number_of_filters_for_temporal_filter:
		previously_filtered_frames = previously_filtered_frames[-number_of_filters_for_temporal_filter:]

	decimation = None # rs.decimation_filter(magnitude = 3.0)
	disparity2depth = rs.disparity_transform(True)
	temporal = rs.temporal_filter(smooth_alpha = 0.760, smooth_delta = 80, persistence_control = 0)
	depth2disparity = rs.disparity_transform(False)
	hole_filling = rs.hole_filling_filter(mode = 1)

	min_index = max(-len(previously_filtered_frames), -number_of_filters_for_temporal_filter)
	for frame in previously_filtered_frames[min_index:]:
		apply_filters(frame, [decimation, disparity2depth, temporal, depth2disparity, hole_filling])

	previously_filtered_frames.append(depth_frame)
	return apply_filters(depth_frame, [decimation, disparity2depth, temporal, depth2disparity, hole_filling]).as_depth_frame()
