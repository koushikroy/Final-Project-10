python train_early_fusion.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 6 --epochs 30 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6"


python train_early_fusion.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 6 --epochs 30 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6"

python Code\early_fusion_2d_cnn\train_early_fusion.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 6 --epochs 30 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6"


python train_early_fusion.py --data_root "path/to/dataset" --num_frames 6 --epochs 30 --batch_size 16 --num_workers 0 --out_dir "path/to/output"


python test_early_fusion.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\best_model.pth" --num_frames 16 --batch_size 10 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6"
