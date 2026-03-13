
# python scripts/pretrain.py --dataset ebai --root_dir ./data/EBAI/raw/ --filenames 'BMS_Data_2026-01-16T14-38-45-380Z.csv,BMS_Data_2026-01-22T11-16-48-168Z.csv' --n_cell 9  --save_dir ./checkpoints/ebai/ --batch_size 256 --epochs 500 --use_balancer True --use_scheduler True --window_length 100 --step 1

# python scripts/pretrain.py --dataset fobss --foldername "all" --n_cell 11  --save_dir ./checkpoints/fobss/ --epochs 50


# python scripts/pretrain.py --dataset ebai --root_dir ./data/EBAI/raw/ --filenames 'BMS_Data_2026-01-16T14-38-45-380Z.csv,BMS_Data_2026-01-22T11-16-48-168Z.csv' --n_cell 9  --save_dir ./checkpoints/ebai/ --batch_size 256 --epochs 500 --use_balancer True --use_scheduler True --window_length 100 --step 1 --prefix pretrain-model-w_bayesian

# python scripts/pretrain.py --dataset ebai --root_dir ./data/EBAI/raw/ --filenames 'BMS_Data_2026-01-16T14-38-45-380Z.csv,BMS_Data_2026-01-22T11-16-48-168Z.csv' --n_cell 9  --save_dir ./checkpoints/ebai/ --batch_size 256 --epochs 500 --use_balancer True --use_scheduler True --window_length 100 --step 1 --prefix pretrain-model-w_o_bayesian

# python scripts/pretrain.py --dataset fobss --foldername "profile_-25A_10A_04_12_18" --n_cell 11 --use_balancer False  --use_scheduler False --save_dir ./checkpoints/fobss/ --epochs 1000  --save_interval 100 --prefix pretrain-model-w_bayesian

# python scripts/pretrain.py --dataset fobss --foldername "profile_-25A_10A_04_12_18" --n_cell 11 --use_balancer False  --use_scheduler False --save_dir ./checkpoints/fobss/ --epochs 1000 --save_interval 100 --prefix pretrain-model-w_o_bayesian


# python scripts/pretrain.py --dataset fobss --foldername "profile_-25A_10A_04_12_18" --n_cell 11 --use_balancer True  --use_scheduler True --save_dir ./checkpoints/fobss/ --epochs 100  --save_interval 5000 --prefix pretrain-model-w_LossBalancer

# python scripts/pretrain.py --dataset fobss --foldername "profile_-25A_10A_04_12_18" --n_cell 11 --use_balancer False  --use_scheduler True --save_dir ./checkpoints/fobss/ --epochs 100  --save_interval 5000 --prefix pretrain-model-w_o_LossBalancer

# python scripts/pretrain.py --dataset ebai --root_dir ./data/EBAI/raw/ --filenames 'BMS_Data_2026-01-16T14-38-45-380Z.csv' --n_cell 9  --save_dir ./checkpoints/ebai/ --batch_size 256 --epochs 200 --use_balancer --use_scheduler --window_length 100 --step 1 --prefix pretrain-model-w_LossBalancer

# python scripts/pretrain.py --dataset ebai --root_dir ./data/EBAI/raw/ --filenames 'BMS_Data_2026-01-16T14-38-45-380Z.csv' --n_cell 9  --save_dir ./checkpoints/ebai/ --batch_size 256 --epochs 200  --use_scheduler --window_length 100 --step 1 --prefix pretrain-model-w_o_LossBalancer