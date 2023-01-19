
## Implementation for COLING2022 Paper: Event Causality Extraction with Event Argument Correlations

### Requirements

* torch              1.7.1+cu110
* transformers       4.5.1

### Data
The raw data we used is available [here](https://tianchi.aliyun.com/dataset/dataDetail?dataId=110901), and our proposed data is available [here](https://github.com/cuishiyao96/ECE/tree/main/data)

* Description to directory of our processed data
```
.
|-processed_data # processed data in the format of pickle
|-split_data     # splits of train, dev and test
|rawdata.txt     
```


### Train and test the model

```
mkdir log
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
@inproceedings{cui-etal-2022-event,
    title = "Event Causality Extraction with Event Argument Correlations",
    author = "Cui, Shiyao  and
      Sheng, Jiawei  and
      Cong, Xin  and
      Li, Quangang  and
      Liu, Tingwen  and
      Shi, Jinqiao",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.201",
    pages = "2300--2312",
    abstract = "Event Causality Identification (ECI), which aims to detect whether a causality relation exists between two given textual events, is an important task for event causality understanding. However, the ECI task ignores crucial event structure and cause-effect causality component information, making it struggle for downstream applications. In this paper, we introduce a novel task, namely Event Causality Extraction (ECE), aiming to extract the cause-effect event causality pairs with their structured event information from plain texts. The ECE task is more challenging since each event can contain multiple event arguments, posing fine-grained correlations between events to decide the cause-effect event pair. Hence, we propose a method with a dual grid tagging scheme to capture the intra- and inter-event argument correlations for ECE. Further, we devise a event type-enhanced model architecture to realize the dual grid tagging scheme. Experiments demonstrate the effectiveness of our method, and extensive analyses point out several future directions for ECE.",
}
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
