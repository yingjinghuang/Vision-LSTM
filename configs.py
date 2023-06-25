import os

configs = {
    # Parameters
    # model name for folders of weights and tensorboard
    "model_name" : "m3_lstm",
    # GPU id(s) to use
    # Format: a comma-delimited list. 0,1,2,3
    "gpu" : "0",
    # batch size
    "batch_size" : 128,
    # num of epochs
    "epochs" : 100,
    # initial learning rate
    "lr" : 0.1,
    # warmup learning rate
    "lr_warm": 0.0035,
    # initial weight decay
    "wd" : 1e-4,
    # number of workers
    "workers" : 1,
    # the mode to decomposite street view
    # Format: single, mean, max, sum, lstm
    "mode" : "lstm", 

    "modality_count": 3,
    "modalities": ["remote", "sv", "mobility"], 

    # Paths
    "weights_folder": "Vision-LSTM//weights",
    "tensorboard_folder": "Vision-LSTM//tensorboard",
    "log_folder": "Vision-LSTM//logs",
    "model_data_path": "Vision-LSTM/data/grids250/model_data.pkl",
    "rs_path": "Vision-LSTM/data/grids250/rs_tiles"
}