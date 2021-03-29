import json
import os
import glob

import pandas as pd

import time
import rosbag_to_videos

error_message_txt = """
The file should contain 12 columns. Please make sure you have not changed any parameters in 
the Elan software. When exporting as TXT, make sure that you have included the time values 
for:
- start time,
- end time
- duration
These values have to be in 3 first formats 
proposed by Elan : 
- 'hh:mm:ss.ms'
-'ss.msec'
-'msec'

This should by the configuration when clicking the export button:
[ ] Limité à l'interval de temps séléctionné
[x] Ajouter l'offset média principal à l'annotation
[x] Include header lines containing media file information
[ ] Exclude tier name from output
[ ] Exclude participant name from output
[ ] Colonne séparée pour chaque piste
[ ] Include description from the controlled vocabulary

Inclure la colonne d'heure pour: | Inclure le format d'heure:
[x] Temps de départ              | [x] hh:mm:ss.ms
[x] temps de fin                 | [x] ss.msec
[x] Durée                        | [x] msec
                                 | [ ] SMPTE Timecode (hh:mm:ss:ff)
"""
error_message_csv = error_message_txt
error_message_json = ""  # Should be able to detect any parameters


def decode_elan_file(content):
	return content.encode("latin-1").decode("utf-8")  # Not sure, but the output of ELAN looks like Latin-1

def decode_filename_elan(original_string:str):
	original_string = original_string.replace("file:///", "")
	original_string = decode_elan_file(original_string)
	return original_string

def read_annotations_as_txt(filename, separator = "\t"):
	video_filename = None
	original_filename = None
	with open(filename, "r") as file:
		oc = file.readline()
		splitted = oc.split(" -- offset:")
		if len(splitted) == 1:
			raise Exception(
				f"We were unable to retrieve the original video filename because the offset information is not "
				f"found on first line of {filename}")
		video_filename = decode_filename_elan(splitted[0][2:])  # Remove the two characters useless here ( #" )
		video_filename = video_filename.replace("file:///", "")
		original_filename = video_filename.replace("_as_color_video", "")

	if original_filename is None:
		raise Exception(f"We were unable to retrieve the original video filename")
	if not os.path.exists(original_filename):
		raise Exception(f"The file {original_filename} doesn't exist")

	content = pd.read_csv(filename, sep = separator, skiprows = [0], header = None)
	print(content)
	if len(content.columns) != 12:
		raise Exception(error_message_txt)

	start_time_column = 3
	end_time_column = 6
	# duration_column = 9
	label_column = 11
	content = content[[start_time_column, end_time_column, label_column]]
	content.columns = ["start", "end", "label"]
	content = pd.concat([content, pd.Series([original_filename] * len(content.values), name="filename")], axis = 1)

	return content

def read_annotations_as_csv(filename):
	return read_annotations_as_txt(filename, separator = ",")

def read_annotations_as_json(filename):
	with open(filename, "r") as file:
		content = json.load(file)
		annotations = pd.DataFrame(data = None, columns = ["start", "end", "label", "filename"])

		for actor in content:
			annotation_per_actor = actor['first']['items']
			for annotation in annotation_per_actor:
				label = annotation['body']['value']

				if 'id' in annotation['target']: # Exported with timing option "#<begin>-<end>" timestamp
					id_annot = annotation['target']['id']
					splitted_id = id_annot.split("#t=")
					video_filename = decode_filename_elan(splitted_id[0])
					timing = splitted_id[-1].split(",") # Time is in seconds
				else:
					video_filename = decode_filename_elan(annotation['target']['source'])
					timing = annotation['target']['selector']['value']
					timing = timing.replace("t=", "")
					timing = timing.split(",")

				original_filename = video_filename.replace("_as_color_video", "")
				start_time = timing[0]
				end_time = timing[1]
				annotations = annotations.append({'filename': original_filename,
				                                  'start': float(start_time),
				                                  'end': float(end_time),
				                                  'label': label}, ignore_index = True)

		return annotations

def update_annotations_timestamps(annotations):
	pass

def create_elan_file(directory:str, elan_filename:str):
	files = glob.glob(directory + "/*_as_color_video_person_*")
	video_file_lines = []
	for file in files:
		video_file_lines.append('<MEDIA_DESCRIPTOR MEDIA_URL="file:///' + os.path.abspath(file) + '" MIME_TYPE="video/mp4"/>')

	print(time.strftime("%Y-%m-%DT%H:%M:%S%z", time.gmtime()))

	file_content = """
<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="LISTIC" DATE="2021-03-24T11:26:58+01:00"
    FORMAT="3.0" VERSION="3.0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"> """ + "\n".join(video_file_lines) + """
    
    <PROPERTY NAME="URN">urn:nl-mpi-tools-elan-eaf:24bb2097-2c69-4d16-99bb-f36fed476ebd</PROPERTY>
        <PROPERTY NAME="lastUsedAnnotationId">0</PROPERTY>
    </HEADER>
    <TIME_ORDER/>
    <TIER LINGUISTIC_TYPE_REF="default-lt" TIER_ID="default"/>
    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false"
        LINGUISTIC_TYPE_ID="default-lt" TIME_ALIGNABLE="true"/>
    <CONSTRAINT
        DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT
        DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT
        DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>
"""

if __name__ == "__main__":
	# print(read_annotations_as_json("annotations/annot3.json"))
	create_elan_file("recordings", "bla.elan")