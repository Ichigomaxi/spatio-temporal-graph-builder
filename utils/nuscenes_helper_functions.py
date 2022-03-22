from typing import List
from nuscenes.nuscenes import NuScenes

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
