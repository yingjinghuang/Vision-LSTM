import os

# model name for folders of weights and tensorboard
model_name = "multi_single_sv"
# GPU id(s) to use
# Format: a comma-delimited list
gpu = "0,1,2,3"
# batch size
batch_size = 128
# num of epochs
epochs = 100
# initial learning rate
lr = 0.1
# initial weight decay
wd = 1e-4
# patience of learning rate
lr_p = 10
# factor of learning rate
lr_f = 0.1
# the ratio of validation dataset
vr = 0.2
# number of workers
workers = 6
# the mode to decomposite street view
# Format: mean
mode = "mean"