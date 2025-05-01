
# Train late fusion frames 6
python Code\cnn_rnn\cnn_resnet_rnn_lstm\train.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 6 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_6\no_augment"

# Test late fusion frames 6
python Code\cnn_rnn\cnn_resnet_rnn_lstm\test.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_6\no_augment\best_model.pth" --num_frames 6 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_6\no_augment"

# Train late fusion frames 6 --augment
python Code\cnn_rnn\cnn_resnet_rnn_lstm\train.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 6 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_6\augment" --augment

# Test late fusion frames 6 --augment
python Code\cnn_rnn\cnn_resnet_rnn_lstm\test.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_6\augment\best_model.pth" --num_frames 6 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_6\augment"

# Train late fusion frames 10
python Code\cnn_rnn\cnn_resnet_rnn_lstm\train.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 10 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_10\no_augment"

# Test late fusion frames 10
python Code\cnn_rnn\cnn_resnet_rnn_lstm\test.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_10\no_augment\best_model.pth" --num_frames 10 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_10\no_augment"

# Train late fusion frames 10 --augment
python Code\cnn_rnn\cnn_resnet_rnn_lstm\train.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 10 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_10\augment" --augment

# Test late fusion frames 10 --augment
python Code\cnn_rnn\cnn_resnet_rnn_lstm\test.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_10\augment\best_model.pth" --num_frames 10 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_10\augment"


# Train late fusion frames 16
python Code\cnn_rnn\cnn_resnet_rnn_lstm\train.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 16 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_16\no_augment"

# Test late fusion frames 16
python Code\cnn_rnn\cnn_resnet_rnn_lstm\test.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_16\no_augment\best_model.pth" --num_frames 16 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_16\no_augment"

# Train late fusion frames 16 --augment
python Code\cnn_rnn\cnn_resnet_rnn_lstm\train.py --data_root "D:\research\deep_learning\Final-Project-10\dataset" --num_frames 16 --epochs 100 --patience 20 --batch_size 16 --num_workers 0 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_16\augment" --augment

# Test late fusion frames 16 --augment
python Code\cnn_rnn\cnn_resnet_rnn_lstm\test.py  --data_root "D:\research\deep_learning\Final-Project-10\dataset" --model_path "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_16\augment\best_model.pth" --num_frames 16 --batch_size 16 --out_dir "D:\research\deep_learning\Final-Project-10\Code\cnn_rnn\cnn_resnet_rnn_lstm\results\frames_16\augment"