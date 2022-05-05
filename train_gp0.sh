# To train the default config must be loaded first.(This is already done in the train.py script)
# To train with other configs just add:  with {path_2_config}
# This will overwrite/update the previously loaded config values.
# As a result you will train with a different configuration

# Default run 
# python train.py
# Multiclass training
python train.py with 'gpu_settings={"device_id": [0] ,"torch_device": "cuda:0"} ' 