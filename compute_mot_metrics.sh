RESULT_PATH="/media/HDD2/students/maximilian/spatio-temporal-gnn/evalutation_single_graphs/05-19_17:46_evaluation_single_graphs/0/mini_val_tracking.json"
# [--output_dir OUTPUT_DIR] 
OUTPUT_DIR='/media/HDD2/students/maximilian/spatio-temporal-gnn/mot_metric/mini_val'
# [--eval_set EVAL_SET] 
EVAL_SET="mini_val"
# [--dataroot DATAROOT] 
DATAROOT='/media/HDD2/Datasets/mini_nusc'
# [--version VERSION] 
VERSION='v1.0-mini'
# [--config_path CONFIG_PATH] 
CONFIG_PATH='.configs/nuscenes_eval/mot_car_evaluation.json'
# [--render_curves RENDER_CURVES] 
RENDER_CURVES=TRUE
# [--verbose VERBOSE] 
VERBOSE=TRUE
# [--render_classes RENDER_CLASSES [RENDER_CLASSES ...]
# $RENDER_CLASSES=None


python /home/maximilian/anaconda3/envs/nuscenes/lib/python3.9/site-packages/nuscenes/eval/tracking/evaluate.py \
            $RESULT_PATH \
            --output_dir $OUTPUT_DIR \
            --eval_set $EVAL_SET \
            --dataroot $DATAROOT \
            --version $VERSION \
            --config_path $CONFIG_PATH \
            --render_curves $RENDER_CURVES \
            --verbose $VERBOSE

        