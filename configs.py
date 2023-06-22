import os

configs = {
    # Parameters
    # model name for folders of weights and tensorboard
    "model_name" : "m3_single_sv",
    # GPU id(s) to use
    # Format: a comma-delimited list. 0,1,2,3
    "gpu" : "0,1,2,3",
    # batch size
    "batch_size" : 128,
    # num of epochs
    "epochs" : 100,
    # initial learning rate
    "lr" : 0.1,
    # initial weight decay
    "wd" : 1e-4,
    # patience of learning rate
    "lr_p" : 10,
    # factor of learning rate
    "lr_f" : 0.1,
    # the ratio of validation dataset
    "vr" : 0.2,
    # number of workers
    "workers" : 6,
    # the mode to decomposite street view
    # Format: single, mean, max, sum, lstm
    "mode" : "mean", 

    "modality_count": 3,
    "modalities": ["remote", "sv", "mobility"], 

    # Paths
    "weights_folder": "./weights",
    "tensorboard_folder": "./tensorboard",
    "log_folder": "./logs",
    "model_data_path": "data/grids/model_data.pkl",
    "rs_path": "Vision-LSTM/data/grids500/rs_tiles"
}