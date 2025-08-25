
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --amp --epoch 1 --cuda --batch_size 1 --resume True --inference  --save_dir results_main/tpenet_swinl/diode/  --candidate_len 30 --p_threshold 0.7 --resume_checkpoint ./savedModels/tpenet_swinl_diode/ --config_path ./config.yml --previous_mask  --val_dataset_name diode

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --amp --epoch 1 --cuda --batch_size 1 --resume True --inference  --save_dir results_main/tpenet_swinl/entity/  --candidate_len 30 --p_threshold 0.7 --resume_checkpoint ./savedModels/tpenet_swinl_entity/ --config_path ./config.yml --previous_mask --previous_mask --val_dataset_name entityseg  
