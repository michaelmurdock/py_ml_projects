# face_detector_google3.py
#
# This version expects a source folder containing folders of images,
# as you get with the LFW distribution.
# Each folder is named for one person and that folder should contain
# only photos of that person.

#
# Environment Variable: GOOGLE_APPLICATION_CREDENTIALS
# C:\pyDev\__My Scripts\face_detector_google\Face-Detection-3dc1b370d617.json
from __future__ import print_function


"""Draws squares around faces in the given image."""

import sys
import os
import os.path

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)
from mcm_lib2 import exception_utils as eu
from mcm_lib2 import fname as fnm
from mcm_lib2 import files_and_folders as ff
from mcm_lib2 import enum as en



import argparse
import base64
import json
import fnmatch


from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
from PIL import Image
from PIL import ImageDraw

import numpy as np

import kairos_face
kairos_face.settings.app_id = "56aab423"
kairos_face.settings.app_key = "faa3e1412c97b3171dd7dcda3382313a"

RADIUS = 2
API_KEY = 'AIzaSyD3HsHlSOrQXhmqjpph9R9Di1pl_4WVNEY'



def get_list_of_matching_files(root_dir, image_ext_tuple):
  '''
  Returns a list of filenames in the specified directory that match the specified tuple.
  Example tuple:   ('*.jpg', '*.jpeg', '*.j2k', '*.png')
  '''
  x_matching_files = []  

  # Read the directories in the root_dir and then iterate over them
  dirs = os.listdir(root_dir) 
  for dir in dirs:
    full_dir = os.path.join(root_dir, dir)
    for root, dirs, files in os.walk(full_dir):
      for extension in image_ext_tuple:
        for filename in fnmatch.filter(files, extension):
          full_filename = os.path.join(full_dir, filename)
          x_matching_files.append(full_filename)

  return x_matching_files

def save_as_json(d, full_json_filename):
  '''
 
  '''
  try:
    json_string = json.dumps(d, indent=4)
         
  except Exception as e:
    return(False, 'Exception in save calling json.dumps. Details: %s' % (str(e)))

  try:
    with open(full_json_filename, "w") as text_file:
      text_file.write("%s" % (json_string))
  except Exception as e:
    return (False, 'Exception in save() writing to output file: %s. Details: %s' % (full_json_filename, str(e)))
  return (True, '')       

def get_elipse_bounding_box(x, y, radius):
  '''
  Returns as a list the bounding box around the specified point
  '''
  #left    = (int(x)-radius, int(y))
  #right   = (int(x)+radius, int(y))
  #top     = (int(x), int(y)-radius)
  #bottom  = (int(x), int(y)+radius)

  top_left = (int(x)-radius, int(y)-radius)
  bottom_right =  (int(x)+radius, int(y)+radius)

  #return [left, right, top, bottom]
  return [top_left, bottom_right]

# [START get_vision_service]
def get_vision_service():
  credentials = GoogleCredentials.get_application_default()
  return discovery.build('vision', 'v1', credentials=credentials)
# [END get_vision_service]


def detect_face(face_file, service, max_results=4):
  '''
  Uses the Vision API to detect faces in the face_file image
  object that was opened by the client with the following:
  with open(input_filename, 'rb') as face_file:
    detect_face(face_file, 3) 

  RETURNS: the following tuple (result_flag, err_msg, response_obj, face_data)
  '''
  # Read the previously-opened image file, base64-encode it and then decode it
  image_content = face_file.read()
  batch_request = [{
    'image': { 'content': base64.b64encode(image_content).decode('utf-8') },
    'features': [{ 'type': 'FACE_DETECTION', 'maxResults': max_results }]
    }]

  #service = get_vision_service()
  # Exception details: <HttpError 403 when requesting https://vision.googleapis.com/v1/images:annotate?alt=json 
  # returned "The request cannot be identified with a client project. Please pass a valid API key with the request.">
  #service = discovery.build('vision', 'v1')
  
  #API_KEY = 'AIzaSyD3HsHlSOrQXhmqjpph9R9Di1pl_4WVNEY'
  #service = discovery.build('vision', 'v1',  developerKey = API_KEY)

  try:
    request = service.images().annotate(body={ 'requests': batch_request })
  except Exceptiion as e:
    msg = 'Exception calling annotate service. Details: %s' % (str(e))
    return (False, msg, None, None)

  try:
    response = request.execute()
  except Exception as e:
    msg = 'Exception calling request.execute. Details: %s' % (str(e))
    return (False, msg, None, None)

  try:
    face_data = None
    if 'faceAnnotations' in response['responses'][0].keys():
      face_data = response['responses'][0]['faceAnnotations']
  except Exception as e:
    msg = 'Exception accessing response object for face_data. Details: %s' % (str(e))
    return (False, msg, response, None)

  return (True, '', response, face_data)

def draw_landmark_boxes(image, xz_landmarks, output_filename):
  '''
  draws a polygon around each landmark and then save out the file to the specified
  output_filename.
  '''
  im = Image.open(image)
  draw = ImageDraw.Draw(im)

  fill='#00ff00'

  for z_landmark in xz_landmarks:
    x = z_landmark['position']['x']
    y = z_landmark['position']['y']
    x_bbox = get_elipse_bounding_box(x, y, RADIUS)
    draw.ellipse(x_bbox, fill=fill)

  im.save(output_filename)

def highlight_faces(image, faces, output_filename):
  '''
    Draws a polygon around the faces, then saves to output_filename.

    Args:
      image: a file containing the image with the faces.
      faces: a list of faces found in the file. This should be in the format
          returned by the Vision API.
      output_filename: the name of the image file to be created, where the
          faces have polygons drawn around them.
  '''
  im = Image.open(image)
  draw = ImageDraw.Draw(im)

  for face in faces:
    box1 = [(v.get('x', 0.0), v.get('y', 0.0)) for v in face['fdBoundingPoly']['vertices']]
    #box2 = [(v.get('x', 0.0), v.get('y', 0.0)) for v in face['boundingPoly']['vertices']]
    draw.line(box1 + [box1[0]], width=3, fill='#00ff00')
    #draw.line(box2 + [box2[0]], width=3, fill='#00ff0f')

  im.save(output_filename)

def detect_and_annotate(input_filename, face_filename, json_filename, service, max_results):
  '''

  RETURNS: the following tuple: (result_flag, base_filename, num_faces, headwear_likelihood, msg)
  '''
  num_faces = 0
  base_filename = os.path.basename(input_filename)
  tmp_output = os.path.join(os.path.dirname(face_filename),'tmp.jpg')

  # First detect the face, then draw a box around it, then save it
  with open(input_filename, 'rb') as source_image:
    (result, err_msg, response, face_data) = detect_face(source_image, service, max_results)
    if not result:
      msg = 'Error in detect_and_annotate calling detect_face. Details: %s' % err_msg
      return (False, base_filename, 0, msg)
 
    # the call didn't return face data
    if not face_data:
      msg = 'No face annotation data returned for %s' % (face_filename)
      return (False, base_filename, 0, msg)


    # The call to detect_face succeeded and we have face_data
    num_faces = len(face_data)

    #print('Found {} face{}'.format(num_faces, '' if  num_faces == 1 else 's'))
    #print('Writing face rectangle to file {}'.format(face_filename))

    # Reset the file pointer, so we can read the file again to draw the face rectangle
    try:
      source_image.seek(0)
      highlight_faces(source_image, face_data, tmp_output)

    except Exception as e:
      msg = 'Exception in highlight_faces. Details: %s' % (str(e))
      return (False, base_filename, num_faces, msg)

  try:
    # Draw ellipses for the landmarks on the image and save it to a different filename
    with open(tmp_output, 'rb') as source_image:
      xz_landmarks = face_data[0]['landmarks']
      draw_landmark_boxes(tmp_output, xz_landmarks, face_filename)

  except Exception as e:
    msg = 'Exception in draw_landmark_boxes. Details: %s' % (str(e))
    return (False, base_filename, num_faces, msg)

  try:
    # Save the JSON file
    (result, errmsg) = save_as_json(response, json_filename)
    if not result:
      return (False, base_filename, num_faces, errmsg)

  except Exception as e:
    msg = 'Exception calling save_as_json. Details: %s' % (str(e))
    return (False, base_filename, num_faces, msg)

  # Iterate over the list of face_data
  xz_face_data = []
  for idx in xrange(0, num_faces):

    z_face_data = {}

    # Pull out headwearLikelihood value from json
    headwear_likelihood = face_data[0]['headwearLikelihood']
    z_face_data['headwear_likelihood'] = face_data[idx]['headwearLikelihood']


    # Pull out eye locations
    eye_data = get_eye_locations(face_data, idx)
    if len(eye_data) == 2:
      try:
        d = compute_eye_distance(eye_data)
      except Exception as e:
        print('Exception calling compute_eye_distance: %s' % (str(e)))
        return (False,  base_filename, xz_face_data, '')

      if d > 0.0:
        z_face_data['eye_distance'] = d
      else:
         z_face_data['eye_distance'] = 0.0
    else:
      z_face_data['eye_distance'] = 0.0

    # Pull out pan angles
    try:
      z_face_data['face_angles'] =  get_face_angles(face_data, idx)
    except Exception as e:
      print('Exception calling get_face_angles: %s' % (str(e)))
      return (False, base_filename, z_face_data, '')

    # Add this dictionary to our list
    xz_face_data.append(z_face_data)

  return (True, base_filename, xz_face_data, '')

def compute_eye_distance(eye_data):
  '''
  eye_data is a list of two lists
  eye_data[0]: [left_eye_x, left_eye_y, left_eye_z]
  eye_data[1]:  [right_eye_x, right_eye_y, right_eye_z]
  '''

  left_eye_x, left_eye_y, left_eye_z = eye_data[0]
  right_eye_x, right_eye_y, right_eye_z = eye_data[1]

  if left_eye_x and right_eye_x:
    x = left_eye_x - right_eye_x
    x = x * x
  else:
    return -1.0

  if left_eye_y and right_eye_y:
    y = left_eye_y - right_eye_y
    y = y * y
  else:
    return -1.0

  if left_eye_z and right_eye_z:
    z = left_eye_z - right_eye_z
    z = z * z
  else:
    return -1.0


  d = np.sqrt(x + y + z)
  return d

def select_faces_to_keep(filename, xz_face_data):
  '''
  xz_face_data is a list of dictionaries.
  Each list item is a dictionary with the following key/values:
  'eye_distance' : distance (float)
  'headwear_likelihood' : likelihood enum (string)
  'face_angles' : dictionary with the following keys: 'pan', 'roll', 'pitch'

  Using the data we use some simple heuristics to determine which faces to keep.

  Rule 1. In an ideal condition, the best face is the one that is relatively much larger than 
  the runner-up and has a pan angle close to 0.

  If both faces are about the same size, then the best face is the one with the pan
  angle closest to 0.
 
  '''

  x_faces_to_keep = []

  x_distances = []
  for z in xz_face_data:
    d = z['eye_distance']
    x_distances.append(d)

  # Get the largest face (largest eye distance) and its index in the list
  idx_of_largest_face = get_index_of_largest_eye_distance(x_distances)
  largest_eye_distance = x_distances[idx_of_largest_face]
  z_angles_of_largest_face = xz_face_data[idx_of_largest_face]['face_angles']
  pan_angle_of_largest_face = abs(z_angles_of_largest_face['pan'])  

  # Null out the largest value so we can get runner-up
  x_distances[idx_of_largest_face] = 0.0

  # Get the runner-up face distance and its index in the list
  idx_of_second_largest_eye_distance = get_index_of_largest_eye_distance(x_distances)
  second_largest_eye_distance = x_distances[idx_of_second_largest_eye_distance]
  z_angles_of_runner_up_face = xz_face_data[idx_of_second_largest_eye_distance]['face_angles']
  pan_angle_of_runner_up    = abs(z_angles_of_runner_up_face['pan'])
       

  # Calculate the relative difference between these two distances
  # This is a float between 0 and 1 in which a larger value indicates a greater relative difference
  try:
    relative_difference = calculate_relative_difference(largest_eye_distance, second_largest_eye_distance)
  except Exception as e:
    print('Exception thrown calling calculate_relative_difference. Details: %s' % (str(e)))
    return []

  rel_face_diff = face_difference(relative_difference)

  face_dir_largest = face_direction(pan_angle_of_largest_face)
  face_dir_runnerup = face_direction(pan_angle_of_runner_up)

  # ---------------------------------------------------------------------------------------------------------------
  # Rules for how we deal with other faces detected:
  # R0: Large relative difference in face size and forward-facing ==> Only keep largest face
  # R1: Medium relative difference in face size, but only largest face is forward-facing ==> Only keep largest face
  # R2: 
  # R3
 
 
  # Rule R0: If much larger and forward-facing,  then only keep the largest face
  if (rel_face_diff.name == 'LARGE' or rel_face_diff.name == 'EXTRA_LARGE') and \
      face_dir_largest.d == 'FORWARD':
    print('%s: Rule-0: relative difference: %s (%f), pan of largest face: %s (%f), pan of runner-up: %s (%f)' % 
          (filename, rel_face_diff.name, relative_difference, face_dir_largest.d, pan_angle_of_largest_face, face_dir_runnerup.d, pan_angle_of_runner_up))

  # Rule R1: If larger, forward-facing face and runner-up is not forward-facing, then keep only largest face
  elif rel_face_diff.name == 'MEDIUM' and \
      face_dir_largest.d == 'FORWARD' and \
      (face_dir_runnerup.d == 'ANGLED' or face_dir_runnerup.d == 'SIDE_VIEW'):
    print('%s: Rule-0: relative difference: %s (%f), pan of largest face: %s (%f), pan of runner-up: %s (%f)' % 
          (filename, rel_face_diff.name, relative_difference, face_dir_largest.d, pan_angle_of_largest_face, face_dir_runnerup.d, pan_angle_of_runner_up))

  # Rule R2: If approx same size faces, largest is forward-facing and runner-up is not forward-facing, keep the face that is forward-facing
  elif (rel_face_diff.name == 'EXTRA_SMALL' or  rel_face_diff.name == 'SMALL') and \
        face_dir_largest.d == 'FORWARD' and \
        (face_dir_runnerup.d == 'ANGLED' or face_dir_runnerup.d == 'SIDE_VIEW'):
    print('%s: Rule-0: relative difference: %s (%f), pan of largest face: %s (%f), pan of runner-up: %s (%f)' % 
          (filename, rel_face_diff.name, relative_difference, face_dir_largest.d, pan_angle_of_largest_face, face_dir_runnerup.d, pan_angle_of_runner_up))

  # Rule R3: If approx same size faces, largest is forward-facing and runner-up is forward-facing, keep both faces
  elif (rel_face_diff.name == 'EXTRA_SMALL' or  rel_face_diff.name == 'SMALL') and \
        face_dir_largest.d == 'FORWARD' and face_dir_runnerup.d  == 'FORWARD':
    print('%s: Rule-0: relative difference: %s (%f), pan of largest face: %s (%f), pan of runner-up: %s (%f)' % 
          (filename, rel_face_diff.name, relative_difference, face_dir_largest.d, pan_angle_of_largest_face, face_dir_runnerup.d, pan_angle_of_runner_up))

  # Rule R4: If approx same size faces and largest face is not forward-facing, runner-up is forward-facing ==> Keep only runner-up face
  elif (rel_face_diff.name == 'EXTRA_SMALL' or  rel_face_diff.name == 'SMALL') and \
       (face_dir_largest.d == 'ANGLED' or face_dir_largest.d == 'SIDE_VIEW') and \
       face_dir_runnerup.d == 'FORWARD':
    print('%s: Rule-0: relative difference: %s (%f), pan of largest face: %s (%f), pan of runner-up: %s (%f)' % 
          (filename, rel_face_diff.name, relative_difference, face_dir_largest.d, pan_angle_of_largest_face, face_dir_runnerup.d, pan_angle_of_runner_up))

  else:
     print('%s: Rule-0: relative difference: %s (%f), pan of largest face: %s (%f), pan of runner-up: %s (%f)' % 
          (filename, rel_face_diff.name, relative_difference, face_dir_largest.d, pan_angle_of_largest_face, face_dir_runnerup.d, pan_angle_of_runner_up))

  return x_faces_to_keep


def get_face_angles(face_data, face_idx):
  '''
  Using the json face_data returned from the Google Vision detection call,
  get the pan, tilt and roll angles of the face and return in a dictionary,
  with these values keyed on the angle name.
  '''
  face_angles = {}
  
  try:
    face_angles['pan']  = face_data[face_idx]['panAngle']
    face_angles['tilt'] = face_data[face_idx]['tiltAngle']
    face_angles['roll'] = face_data[face_idx]['rollAngle']
  except Exception as e:
    face_angles['pan']  = None
    face_angles['tilt'] = None
    face_angles['roll'] = None

  return face_angles

def get_location_from_landmark_dict(z_lm):
  '''
  z_lm is the landmark dictionary and z['position'] is the dictionary
  holding the coordinate values. It appears that sometimes the Google
  service doesn't return a full dictionary.
  '''
  x = None
  if 'x' in z_lm['position'].keys():
    x = z_lm['position']['x']

  y = None
  if 'y' in z_lm['position'].keys():
    y = z_lm['position']['y']

  z = None
  if 'z' in z_lm['position'].keys():
    z = z_lm['position']['z']

  return [x, y, z]
  

def get_eye_locations(face_data, face_idx):
  '''
  Using the json face_data returned from Google Vision detection call,
  get the location of the left and right eye for the specified face index.
  face_idx:0 is the 0th face detected.
  face_idx:1 is the 1st face detected. ...
  '''

  eye_data = [[], []]

  # face_data[idx]['landmarks'] is a list of dictionaries.
  # We iterate over the list looking for the one that has the value of LEFT_EYE or 
  # RIGHT_EYE for the key 'type'.
  for lm in face_data[face_idx]['landmarks']:
  
    if lm['type'] == 'LEFT_EYE':
      #left_eye_x = lm['position']['x']
      #left_eye_y = lm['position']['y']
      #left_eye_z = lm['position']['z']
      #eye_data[0] = [left_eye_x, left_eye_y, left_eye_z]
      [left_eye_x, left_eye_y, left_eye_z] = get_location_from_landmark_dict(lm)
      continue
    
    if lm['type'] == 'RIGHT_EYE':
      #right_eye_x = lm['position']['x']
      #right_eye_y = lm['position']['y']
      #right_eye_z = lm['position']['z']
      #eye_data[1] = [right_eye_x, right_eye_y, right_eye_z]
      [right_eye_x, right_eye_y, right_eye_z] = get_location_from_landmark_dict(lm)
      continue

  return [[left_eye_x, left_eye_y, left_eye_z],  [right_eye_x, right_eye_y, right_eye_z]]

def get_index_of_largest_eye_distance(x_distances):
  '''
  Returns the index in the x_distances list containing the maximum value.
  '''
  max_value = max(x_distances)
  max_index = x_distances.index(max_value)
  return max_index

def calculate_relative_difference(max_distance, runner_up_distance):
  '''
  Returns the relative difference between the max distance and the runner-up:
  Rel_Diff = (max_distance - runner_up) / max_distance
  '''
  return (max_distance - runner_up_distance) / max_distance

class face_direction(object):

  def __init__(self, pan_angle):
    self.pan_angle = pan_angle

    if pan_angle <= 30:
      self._direction = en.cenum(0, 'FORWARD')
    elif pan_angle > 30 and pan_angle <= 80:
      self._direction = en.cenum(1, 'ANGLED')
    else:
      self._direction = en.cenum(2, 'SIDE_VIEW')

  @property
  def d(self):
    return self._direction.name


class face_difference(object):

  def __init__(self, rel_diff):
    '''
    rel_diff is the relative face size difference and is on (0, 1).
    '''
    self.rel_diff = rel_diff
    if rel_diff <= 0.08:
      self.category = en.cenum(0, 'EXTRA_SMALL')
    elif rel_diff > 0.08 and rel_diff <= 0.12:
      self.category = en.cenum(2, 'SMALL')
    elif rel_diff > 0.12 and rel_diff <= 0.20:
      self.category = en.cenum(3, 'MEDIUM')
    elif rel_diff > 0.20 and rel_diff <= 0.60:
      self.category = en.cenum(4, 'LARGE')
    else:
      self.category = en.cenum(5, 'EXTRA_LARGE')
    
  @property
  def name(self):
    return self.category.name



def create_exclude_list(exclude_filename):
  '''
  Returns a list of filenames in the exclude_filename file.
  These are the filenames that should be exluded from processing.
  '''
  with open(exclude_filename) as f:
    x_names = [line.strip() for line in f]
  return x_names


if __name__ == '__main__':

  # Visual Studio script arguments:
  # tst1\00AB500A-0006-0000-0000-000000000000.jpg --out 00AB500A-0006-0000-0000-000000000000_out.jpg --max-results 5
  # tst1\demo-image.jpg --out tst1\dog_out.jpg --max-results 3

  # tst1\00AB500A-0006-0000-0000-000000000000.jpg --face 00AB500A-0006-0000-0000-000000000000_face.jpg --land 00AB500A-0006-0000-0000-000000000000_land.jpg --max-results 5
  # tst1\02ED2000-0006-0000-0000-000000000000.jpg --face 02ED2000-0006-0000-0000-000000000000_face.jpg --land 02ED2000-0006-0000-0000-000000000000_land.jpg --max-results 5


  # fd = face_difference(0.30)
  # print(fd.name)
  
  print(sys.prefix)
  print(sys.version)
  print(sys.path)

  src_root_dir  = r'E:\_Ancestry\lfw\lfw_tmp_efghijk_orig'
  out_dir       = r'E:\_Ancestry\lfw\lfw_output'
  out_suffix    = '_face'

  #exclude_list_filename = 'exclude1.txt'
  #x_exclude = create_exclude_list(exclude_list_filename)

  service = discovery.build('vision', 'v1',  developerKey = API_KEY)

  id = 0
  x_files = get_list_of_matching_files(src_root_dir, ('*.jpg', '*.jpeg'))
  #for fn in x_files:
    
  #  input_face_fni = fni.fname_info(fullname=fn)

  #  basename = 
  #  output_face_fni = fni.fname_info(dirname=out_dir, basename=input_face_fni.basename, suffix=out_suffix)

  #  (dir, filename) = os.path.split(fn)
  #  (basename, ext) = os.path.splitext(filename)

  for fn in x_files:

    #if fn in x_exclude:
    #  msg = '%s | %s' % ('Exclude', fn)
    #  print(msg)
    #  continue
   
    # Name of the output image file (with the out_suffix)
    face_fn = basename + out_suffix + '.jpg'
    full_output_face_fn = os.path.join(out_dir, face_fn)

    # Name of the output JSON file (.json ext)
    json_fn = basename + '.json'
    full_output_json_fn = os.path.join(out_dir, json_fn)


    
    try:
      (result, base_filename, xz_face_data, errmsg) = detect_and_annotate(fn, full_output_face_fn, full_output_json_fn, service, 3)
    except Exception as e:
      print('Exception calling detect_and_annotate on: %s, Details: %s' % (filename, str(e)))
      continue
    if not result:
      google_result = 'Failure'
    else:
      google_result = 'Success'

      num_faces = len(xz_face_data)

      if num_faces == 1:
        try:
          (response_code, z_attributes) = kairos_face.enroll_face(id, 'gallery13', file=fn)
          kairos_result = 'Success'
        except Exception as e:
          msg = 'Exception in enroll_face for %s. Details: %s' % (basename, str(e))
      
        face_idx = 0
        gender = z_attributes['gender']['type']
        age = z_attributes['age']
        confidence = z_attributes['confidence']

        headwear_likelihood = xz_face_data[0]['headwear_likelihood']
        eye_distance = xz_face_data[0]['eye_distance']
        pan_angle = xz_face_data[0]['face_angles']['pan']

        #pan_angle = xz_face_angles[0]['pan']
        #eye_distance = x_distances[0]

        msg = '%s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s' % (google_result, kairos_result, basename, face_idx, headwear_likelihood, gender, age, confidence, str(pan_angle), str(eye_distance), fn, errmsg)
        print(msg)
        id += 1

      # More than 1 face slightly complicates things ...
      else:

        # We only care about the "extra" face if it meets certain conditions ...
        try:
          x_faces_to_keep = select_faces_to_keep(base_filename, xz_face_data)
        except Exception as e:
          print('Exception in select_faces_to_keep on %s. Details: %s' % (base_filename, str(e)))
          continue

        if False:

          # Iterate over the faces we are going to keep...
          for face_idx in xrange(0, num_faces):

            try:
              (response_code, z_attributes) = kairos_face.enroll_face(id, 'gallery13', file=fn)
              kairos_result = 'Success'
            except Exception as e:
              msg = 'Exception in enroll_face for %s. Details: %s' % (basename, str(e))
      
            gender = z_attributes['gender']['type']
            age = z_attributes['age']
            confidence = z_attributes['confidence']



            msg = '%s | %s | %s | %d | %s | %s | %s | %s | %s | %s' % (google_result, kairos_result, basename, face_idx, headwear_likelihood, gender, age, confidence, fn, errmsg)
            print(msg)
            id += 1

    #  if num_faces == 1:
    #    try:
    #      (response_code, z_attributes) = kairos_face.enroll_face(id, 'gallery13', file=fn)
    #      kairos_result = 'Success'
    #    except Exception as e:
    #      msg = 'Exception in enroll_face for %s. Details: %s' % (basename, str(e))
      
    #    gender = z_attributes['gender']['type']
    #    age = z_attributes['age']
    #    confidence = z_attributes['confidence']

    #  else:
    #    gender  = 'UNKNOWN_DUE_TO_MULTIPLE_FACES'
    #    age     = 'UNKNOWN_DUE_TO_MULTIPLE_FACES'
    #    confidence = 'UNKNOWN_DUE_TO_MULTIPLE_FACES'

    #msg = '%s | %s | %s | %d | %s | %s | %s | %s | %s | %s' % (google_result, kairos_result, basename, num_faces, headwear_likelihood, gender, age, confidence, fn, errmsg)
    #print(msg)
    #id += 1

  





  #parser = argparse.ArgumentParser(description='Detects faces in the given image.')
  #parser.add_argument('input_image',                                                help='the image you\'d like to detect faces in.')
  #parser.add_argument('--face',         dest='face_output', default='face.jpg',     help='the name of the face output file.')
  #parser.add_argument('--land',         dest='land_output', default='face.jpg', help='the name of the landmark output file.')
  #parser.add_argument('--max-results',  dest='max_results', default=4,              help='the max results of face detection.')
  #args = parser.parse_args()

  #main(args.input_image, args.face_output, args.land_output, args.max_results)

  print('Done!')