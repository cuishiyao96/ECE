
## Implementation for COLING2022 Paper: Event Causality Extraction with Event Argument Correlations

### Requirements

* torch              1.7.1+cu110
* transformers       4.5.1

### Data
The raw data we used is available [here](https://tianchi.aliyun.com/dataset/dataDetail?dataId=110901), and our proposed data is available [here](https://github.com/cuishiyao96/ECE/tree/main/data)

* Description to directory of our proposed data
```
.
|-processed_data # processed data in the format of pickle
|-split_data     # splits of train, dev and test
|rawdata.txt     
```


### Train and test the model

```
cd ./src/
```

* train (Other parameters have been set as default.)
```
CUDA_VISIBLE_DEVICES=1 python train.py --task_name ece_task  --training 1 --debug 0 --hidden_size 768
```

* Inference
```
CUDA_VISIBLE_DEVICES=1 python train.py --task_name ece_task  --training 0 --debug 0 --hidden_size 768 --model_name model_name
```


### Cite

Please cite our paper as 
```
```

and our used dataset as
```
@misc{tianchi2021EventCausal,
title={CCKS2021 The Dataset for Financial Event and Causal Relation Extraction}, 
url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=110901},
author={Tianchi},
year={2021},
}
```