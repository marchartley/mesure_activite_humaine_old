import os
import sys

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    openpose_dir = dir_path + '/openpose'
    openpose_models_dir = openpose_dir + '/models'
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append(openpose_dir + '/bin/python/openpose/Release'); # Windows
    sys.path.append(openpose_dir)
    os.environ['PATH']  = os.environ['PATH'] + ';' + openpose_dir + '/x64/Release;' +  openpose_dir + '/bin;'
    try:
        from pyopenpose import *
    except:
        pass
        # from openpose import *
        # import openpose
        # print(openpose)
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e