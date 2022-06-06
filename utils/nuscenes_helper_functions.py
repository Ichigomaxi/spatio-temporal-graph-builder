from typing import List
from nuscenes.nuscenes import NuScenes
import numpy as np
import torch
from datasets.nuscenes.classes import ALL_NUSCENES_CLASS_NAMES, id_from_name
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.nuscenes import Box

def get_all_samples_2_list(nusc:NuScenes, scene_token:str) -> List[str]:
    # init List
    sample_list=[]
    # get scene
    scene = nusc.get('scene', scene_token)

    #Iterate over scene
    current_token = scene['first_sample_token']
    sample_list.append(current_token)
    while(current_token != scene['last_sample_token']):
        temp_sample = nusc.get('sample', current_token)
        temp_token = temp_sample['next']
        current_token = temp_token
        sample_list.append(current_token)
    return sample_list

def print_all_sample_numbers(nusc:NuScenes) -> None:
    #Get set of scenes
    scenes = nusc.scene
    for scene in scenes:
        current_token = scene['first_sample_token']
        count_samples = 1
        while(current_token != scene['last_sample_token']):
            temp_sample = nusc.get('sample', current_token)
            temp_token = temp_sample['next']
            current_token = temp_token
            count_samples += 1

        print("Scene:",scene['name'],"has Sample number:",count_samples)

def get_all_sample_annotations_for_one_scene(nusc:NuScenes, scene_token:str) -> List[str]:
    # Init List
    annotation_list=[]
    # Get Scene
    scene = nusc.get('scene', scene_token)
    # Get Annotation Tokens
    current_token = scene['first_sample_token']
    while(current_token != scene['last_sample_token']):
        #Get sample dict
        current_sample = nusc.get('sample', current_token)
        #Get list of annotations for current sample
        temp_annotation_list = current_sample['anns']
        
        for annotation in temp_annotation_list:
            if not (annotation in annotation_list):
                annotation_list.append(annotation)
        # Note next sample for iteration
        temp_token = current_sample['next']
        current_token = temp_token
    
    return annotation_list

def get_filtered_instances_for_one_scene(nusc:NuScenes, scene_token:str, categoryQuery:str='vehicle.car') -> List[str]:
    
    instance_tokens = []

    # Get Scene
    scene = nusc.get('scene', scene_token)

    sample_annotation_tokens = get_all_sample_annotations_for_one_scene(nusc,scene_token=scene_token)

    for sample_annotation_token in sample_annotation_tokens:
        sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
        category = sample_annotation['category_name']
        #Filter out non car tokens
        if( category.find(categoryQuery) != -1 ):
            instance_token = sample_annotation['instance_token']
            #Filter out tokens that are already added to the list
            # Only add each unique token once 
            if not (instance_token in instance_tokens):
                instance_tokens.append(instance_token)
                    
        
    
    return instance_tokens

def is_valid_box(box , center:np.ndarray,spatial_shift_timeframes, num_frames = 3 ):
    offset = 0
    for frame_i in range(num_frames):
        reference_center = box.center + np.array([0,0,offset])
        
        if np.equal(reference_center ,center).all():
            return True
        offset += spatial_shift_timeframes
    
    return False

def is_valid_box_torch(box , center:torch.Tensor,spatial_shift_timeframes:int, device:torch.device, num_frames:int):
    offset = 0
    t_box_center = torch.from_numpy(box.center).to(device)
    center = center.to(device)
    for frame_i in range(num_frames):
        reference_center = t_box_center + torch.tensor([0,0,offset]).to(device)

        if torch.equal(reference_center ,center):
            return True
        offset += spatial_shift_timeframes
    
    return False

def determine_class_id(box_name:str):
    """
    Match the official nuscenes detection object class names:
        e.g vehicle.car, vehicle.bus, ...
    with the official nuscenes tracking class names: 
        e.g. car, bus, pedestrian, truck, motorcycle, ...
    """
    class_id = None
    for tracking_class_name in ALL_NUSCENES_CLASS_NAMES:
        if tracking_class_name in box_name:
            class_id = id_from_name(tracking_class_name)
            return class_id
    assert False, "Given Box_name does not contain a suitable class for the tracking challenge!\n"\
            + "Given detection class name: {} \n".format(box_name)

def skip_sample_token(sample_token:str, num_skip:int ,nuscenes_handle:NuScenes):
    '''
    Return sample token after skipping num_skip tokens.\n
    To return the next token set num_skip = 0
    '''
    assert num_skip >= 0
    next_sample_token = sample_token
    for i in range(num_skip + 1):
        sample = nuscenes_handle.get('sample',next_sample_token)
        # only return sample_token until the end of the sequence/scene
        if(sample['next'] != ''): 
            next_sample_token = sample['next']
    return next_sample_token

def get_all_samples_from_scene(scene_token:str,
                                nuscenes_handle:NuScenes)\
                                -> List[str]:
    samples =[]
    scene = nuscenes_handle.get("scene",scene_token)
    sample_token = scene['first_sample_token']
    samples.append(sample_token)
    while (sample_token != scene['last_sample_token']):
        sample_token = skip_sample_token(sample_token,0, nuscenes_handle=nuscenes_handle)
        samples.append(sample_token)
    assert samples[-1] == scene['last_sample_token']
    return samples

def get_sample_data_table(nuscenes_handle:NuScenes,
                                sensor_channel:str, 
                                sample_token:str):
    '''
    sensor_channel :  e.g. : 'LIDAR_TOP'    
    '''

    sample = nuscenes_handle.get('sample', sample_token)
    # get Sample Data token
    ref_sd_token = sample['data'][sensor_channel]
    # Get Sample Data table
    ref_sd_record = nuscenes_handle.get('sample_data', ref_sd_token)

    return ref_sd_record

def get_sensor_pose(nuscenes_handle:NuScenes,
                                sensor_channel:str, 
                                sample_token:str):

    ref_sd_record = get_sample_data_table(nuscenes_handle, sensor_channel, sample_token)

    # Get Calibrated Sensor table for 
    current_cs_record = nuscenes_handle.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
    # Get Sensor table for 
    sensor_record = nuscenes_handle.get('sensor', current_cs_record['sensor_token'])

    assert sensor_channel == sensor_record['channel'], "Is not the same channel! :\n Given: {} \n Expected {}".format(sensor_record, sensor_channel)
    
    translation_sensor_from_ego_frame:List[float] = current_cs_record['translation']
    rotation_sensor_from_ego_frame = Quaternion(current_cs_record['rotation'])

    return translation_sensor_from_ego_frame, rotation_sensor_from_ego_frame

def get_ego_pose(nuscenes_handle:NuScenes,
                                sensor_channel:str, 
                                sample_token:str):

    # Get ego pose table
    ref_sd_record = get_sample_data_table(nuscenes_handle, sensor_channel, sample_token)
    ref_pose_record = nuscenes_handle.get('ego_pose', ref_sd_record['ego_pose_token'])

    # TODO transformation
    # Homogeneous transformation matrix from global to _current_ ego car frame.
    translation_ego_from_world_frame:List[float] = ref_pose_record['translation']
    rotation_ego_from_world_frame = Quaternion(ref_pose_record['rotation'])

    return translation_ego_from_world_frame, rotation_ego_from_world_frame


def get_sensor_2_ego_transformation_matrix(nuscenes_handle:NuScenes,
                                sensor_channel:str, 
                                sample_token:str,
                                inverse = False) -> np.ndarray:
    '''
    sensor_channel :  e.g. : 'LIDAR_TOP'
    '''
    # TODO transformation
    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    translation_sensor_from_ego_frame, rotation_sensor_from_ego_frame =\
        get_sensor_pose(nuscenes_handle,sensor_channel, sample_token)

    sensor_2_ego_transformation_matrix = \
            transform_matrix(translation_sensor_from_ego_frame, 
                            rotation_sensor_from_ego_frame,
                            inverse = inverse)

    return sensor_2_ego_transformation_matrix

def get_ego_2_world_transformation_matrix(nuscenes_handle:NuScenes,
                                sensor_channel:str, 
                                sample_token:str,
                                inverse = False) -> np.ndarray:
    '''
    sensor_channel :  e.g. : 'LIDAR_TOP'
    '''
    # TODO transformation
    # Homogeneous transformation matrix from global to _current_ ego car frame.
    translation_ego_from_world_frame, rotation_ego_from_world_frame = \
            get_ego_pose(nuscenes_handle,sensor_channel, sample_token)

    ego_2_world_transformation_matrix = \
            transform_matrix(translation_ego_from_world_frame, 
                rotation_ego_from_world_frame,
                inverse= inverse)
    
    return ego_2_world_transformation_matrix

def homogeneous_transformation(transformation_matrix:np.ndarray, 
                                translation: List[float] ,
                                rotation :Quaternion):
    translation.append(1.0)
    np_translation = np.asarray(translation)
    np_transformed_translation:np.ndarray = np_translation
    # transform
    np_transformed_translation = transformation_matrix @ np_translation
    # Make non-homogenous again
    transformed_translation: List[float] = np_transformed_translation.tolist()
    transformed_translation.pop()

    # Transform rotation
    rotation_matrix = transformation_matrix[:3, :3]
    orientation_matrix :np.ndarray = rotation.rotation_matrix
    orientation_matrix = rotation_matrix @ orientation_matrix
    transformed_rotation:Quaternion = Quaternion(matrix=orientation_matrix)

    return transformed_translation, transformed_rotation

def transform_detections_lidar2world_frame(nuscenes_handle:NuScenes, 
                    translation: List[float], orientation: Quaternion , 
                    sample_token:str, 
                    sample_annotation_token:str= None):
    """
    Returns and transforms given translation and rotation from LIDAR_TOP frame into World frame
    If sample_annotation is given a assertion test can be done. However it is not necessary
    """
    absolute_error_threshold = 1e-10
    transformed_translation: List[float] = None
    transformed_rotation: Quaternion = None

    ref_channel = 'LIDAR_TOP'
    lidar_2_ego_transformation_matrix:np.ndarray = \
        get_sensor_2_ego_transformation_matrix(nuscenes_handle, ref_channel,
            sample_token , inverse = False)
    ego_2_world_transformation_matrix:np.ndarray = \
        get_ego_2_world_transformation_matrix(nuscenes_handle, ref_channel, 
            sample_token , inverse = False)
    
    # Transform translation 
    # Make homogeneous 
    translation.append(1.0)
    np_translation = np.asarray(translation)
    np_transformed_translation:np.ndarray = np_translation
    # transform
    np_transformed_translation = lidar_2_ego_transformation_matrix @ np_translation
    np_transformed_translation = ego_2_world_transformation_matrix @ np_transformed_translation
    # Make non-homogenous again
    transformed_translation: List[float] = np_transformed_translation.tolist()
    transformed_translation.pop()

    # Transform rotation
    lidar_2_ego_rotation_matrix = lidar_2_ego_transformation_matrix[:3, :3]
    ego_2_world_rotation_matrix = ego_2_world_transformation_matrix[:3, :3]
    orientation_matrix :np.ndarray = orientation.rotation_matrix
    orientation_matrix = lidar_2_ego_rotation_matrix @ orientation_matrix
    orientation_matrix = ego_2_world_rotation_matrix @ orientation_matrix
    transformed_rotation:Quaternion = Quaternion(matrix=orientation_matrix)

    # Double check correctness if possible
    if sample_annotation_token is not None:
        translation_world_frame, rotation_world_frame = \
                get_gt_sample_annotation_pose(nuscenes_handle,sample_annotation_token)
        #TODO Assertion
        comparison_array = (np.asarray(transformed_translation) - np.asarray(translation_world_frame)) 
        absolute_error = np.linalg.norm(comparison_array)
        assert absolute_error < absolute_error_threshold, 'Translation was not transformed correctly into world coordinates'
        absolute_distance = (rotation_world_frame.absolute_distance(rotation_world_frame,transformed_rotation))
        assert (absolute_distance < absolute_error_threshold), 'Rotation was not transformed correctly into world coordinates'

    return transformed_translation, transformed_rotation

def get_gt_sample_annotation_pose(nuscenes_handle:NuScenes,
                    sample_annotation_token:str= None):
    """
    Returns translation and rotation from given sample_annotation
    Returns:
    translation_world_frame: List[float]
    orientation_world_frame: Quaternion 
    """
    translation_world_frame: List[float] = None
    orientation_world_frame: Quaternion = None

    sample_annotation_table = nuscenes_handle.get("sample_annotation", sample_annotation_token)
    translation: List[float] = sample_annotation_table["translation"]
    orientation: List[float] = sample_annotation_table["rotation"]

    translation_world_frame = translation
    orientation_world_frame: Quaternion = Quaternion(orientation)

    assert translation_world_frame is not None and orientation_world_frame is not None

    return translation_world_frame, orientation_world_frame

def transform_boxes_from_world_2_ego(boxes:List[Box], nuscenes_handle:NuScenes, sensor_channel:str,sample_token):
    translation, rotation = get_ego_pose(nuscenes_handle, 
                            sensor_channel=sensor_channel,
                            sample_token=sample_token)
    for box in boxes:
        # Move box to ego vehicle coord system.
        box.translate(-np.array(translation))
        box.rotate(rotation.inverse)

def transform_boxes_from_world_2_sensor(boxes:List[Box], 
                            nuscenes_handle:NuScenes, 
                            sensor_channel:str,
                            sample_token:str):
    sensor_2_ego_transform = get_sensor_2_ego_transformation_matrix(nuscenes_handle,sensor_channel,sample_token,inverse=True)
    
    ego_2_world_transform = get_ego_2_world_transformation_matrix(nuscenes_handle,sensor_channel,sample_token,inverse=True)

    
    translation_ego, rotation_ego = get_ego_pose(nuscenes_handle, 
                            sensor_channel=sensor_channel,
                            sample_token=sample_token)
    translation_sensor, rotation_sensor = get_sensor_pose(nuscenes_handle, 
                            sensor_channel=sensor_channel,
                            sample_token=sample_token)
    # Set translation to negative
    translation_ego = -np.array(translation_ego)
    translation_sensor = -np.array(translation_sensor)
    # Invert rotations
    rotation_ego = rotation_ego.inverse
    rotation_sensor = rotation_sensor.inverse
    for box in boxes:
        center = box.center.copy().tolist()
        rotation_matrix_inSE3 = box.orientation.transformation_matrix.copy()

        # Move box to ego vehicle coord system.
        box.translate(translation_ego)
        box.rotate(rotation_ego)
        # Validate with own calculation
        center.append(1.0)
        center_homogeneous:np.ndarray = np.asarray(center)
        center_homogeneous = ego_2_world_transform @ center_homogeneous
        rotation_matrix_inSE3 = ego_2_world_transform @ rotation_matrix_inSE3
        assert (np.sum(center_homogeneous[:3] - box.center)<1e-6).all()
        rotation_absolute_distance = (box.orientation.absolute_distance(box.orientation,Quaternion(matrix=rotation_matrix_inSE3)))
        assert ((rotation_absolute_distance)<1e-5)
        
        #  Move box to sensor coord system.
        box.translate(translation_sensor)
        box.rotate(rotation_sensor)

        center_homogeneous = sensor_2_ego_transform @ center_homogeneous
        rotation_matrix_inSE3 = sensor_2_ego_transform @ rotation_matrix_inSE3
        assert (np.sum(center_homogeneous[:3] - box.center)<1e-6).all()
        rotation_absolute_distance = (box.orientation.absolute_distance(box.orientation,Quaternion(matrix=rotation_matrix_inSE3)))
        assert ((rotation_absolute_distance)<1e-5)

