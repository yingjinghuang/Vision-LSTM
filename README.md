# Vision-LSTM
Although street-level imagery has emerged as a valuable tool for observing large-scale urban spaces with unprecedented detail, a more comprehensive and representative approach is required to capture the complexity and diversity of urban environments at different spatial scales.

To address this issue, we propose a deep learning-based module called **Vision-LSTM**, which can effectively obtain vector representation from **varying numbers of street-level images** in spatial units. 

![Framework of Vision-LSTM](./img/Vision-LSTM.png)

## Table of Contents
* [Features](#features)
* [Results](#results)
* [Setup](#setup)
* [Usage](#usage)


## Features
- A multimodal data fusion model (satellite imagery, street view imagery, human
mobility) is proposed
- Visual information and dynamic mobility information are both vital in urban
village recognition
- The Vision-LSTM module is proposed to extract visual features from a varying
number of street images
- The proposed method achieved an overall accuracy of 91.6% in identifying
urban villages

## Results
In our urban village recognition case, the results can be seen in the following table.

| Method                                   | OA(%)     | Kappa     | F1        |
|------------------------------------------|-----------|-----------|-----------|
| No fusion (random image)                 | 88.1      | 0.634     | 0.708     |
| Average Pooling                          | 89.1      | 0.656     | 0.727     |
| Maximum Pooling                          | 79.3      | 0.461     | 0.588     |
| Element-wise Sum                         | 77.4      | 0.432     | 0.566     |
| **Vision-LSTM (proposed in this study)** | **91.6**  | **0.720** | **0.773** |

## Setup
Will update with a requirement file.

## Usage
**Step 1**. Prepare your own datasets.

**Step 2**. Run the [preprocess.py](preprocess.py) to preprocess data.
```bash
python preprocess.py
```

**Step 3**. Revise the configs in [configs.py](configs.py).

**Step 4**. Train your own model.
```bash
python train.py
```

## Citation
Will be updated when it is accepted.