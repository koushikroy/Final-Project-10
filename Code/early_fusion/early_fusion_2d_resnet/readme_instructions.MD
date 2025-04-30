
# Train early fusion frames 6
python Code\early_fusion_2d_cnn\train_early_fusion.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 6 --epochs 100 --patience 50 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6"

# Test early fusion frames 6
python Code\early_fusion_2d_cnn\test_early_fusion.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6\best_model.pth" --num_frames 6 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_6"

# Train early fusion frames 10
python Code\early_fusion\early_fusion_resnet\train_early_fusion_2.py --data_root "D:\research\deep_learning\dataset" --num_frames 10 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion\early_fusion_resnet\results\frames_10"

# Test early fusion frames 10
python Code\early_fusion_2d_cnn\test_early_fusion_2.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_10\best_model.pth" --num_frames 10 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\early_fusion_2d_cnn\results\frames_10"