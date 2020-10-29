# Source code of IncreProtoNet Model for Incremental Few-shot Relation Classification
Inplementation of Our paper "A Two-phase Prototypical Network Model for Incremental Few-shot
Relation Classification" in COLING 2020

## Requirements 

* `pytorch = 1.0.1`
* `json`
* `argparse`
* `sklearn`
* `transformers`

## Model Architecture

![image](https://github.com/betterAndTogether/IncreProtoNet/blob/main/model.png)

## Usage 

### Preparing Dataset and Pretrain files

Due to the large size of glove files (pre-trained word embeddings).
Please download `pretrain.tar` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/58f57bda00eb40be8d10/?dl=1) 
and put it under the data directory. Then run `tar xvf pretrain.tar` to decompress it.

### BaseModel Training

**Glove Embedding Command**
```bash
 python3 train_demo.py --dataset fewrel --embedding_type glove --lr 1e-1
```
**Bert Embedding Commain**
```bash
 python3 train_demo.py --dataset fewrel --embedding_type bert --lr 1e-2
```

**Result**

|embedding_type| learn_rate | pl_weight| train_iter |  ACC    | 
|:------------:|:---------: | :-------:| :---------:| :------:| 
| glove        |    1e-1    |   1e-1   |    25000   |  78.23  |
| bert         |    1e-2    |   1e-1   |    8000    |  88.45  |


### Incremental Model Training 

The code will be upload soon!
