CUDA_VISIBLE_DEVICES=0 python3 main.py --amp --epoch 13 --cuda --save_dir results_main/tpenet_swinl --candidate_len 30  --config_path ./config.yml --previous_mask  # default use scribbles # --matlab_nms 

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --amp --epoch 1 --cuda --batch_size 1 --resume True --inference  --save_dir results_main/tpenet_swinl/diode/  --candidate_len 30 --p_threshold 0.7 --resume_checkpoint ./results_main/tpenet_swinl/model/ --config_path ./config.yml --previous_mask  --val_dataset_name diode

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --amp --epoch 1 --cuda --batch_size 1 --resume True --inference  --save_dir results_main/tpenet_swinl/entity/  --candidate_len 30 --p_threshold 0.7 --resume_checkpoint ./results_main/tpenet_swinl/model/ --config_path ./config.yml --previous_mask --previous_mask --val_dataset_name entityseg
