from typing import List
from nuscenes.nuscenes import NuScenes
import numpy as np

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

def is_valid_box(box , center, num_frames = 3, spatial_shift_timeframes = 20):
    offset = 0
    for frame_i in range(num_frames):
        reference_center = box.center + np.array([0,0,offset])
        # if reference_center.base is not None:
        #     print('Base:',reference_center.base)
        if np.equal(reference_center ,center).all():
            return True
        offset += spatial_shift_timeframes
    
    return False