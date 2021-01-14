# Source code of IncreProtoNet Model for Incremental Few-shot Relation Classification
Inplementation of Our paper "A Two-phase Prototypical Network Model for Incremental Few-shot
Relation Classification" in COLING 2020

## Requirements 

* `pytorch = 1.0.1`
* `json`
* `argparse`
* `sklearn`
* `transformers==3.0.2`

## Model Architecture

![image](https://github.com/betterAndTogether/IncreProtoNet/blob/main/model.png)

## Usage 

### Preparing Dataset and Pretrain files

Due to the large size of glove files (pre-trained word embeddings).
Please download `pretrain.tar` from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/58f57bda00eb40be8d10/?dl=1) 
and put it under the data directory. Then run `tar xvf pretrain.tar` to decompress it.

### BaseModel Training
Open the directory "IncreProtoNet/BaseModel", and run the python file "train_demo.py". 
The saved model are in the directory "IncreProtoNet/BaseModel/checkpoint/". 
The specific commands and several important hyper-parameters are as follows:

**Glove Embedding Command**
```bash
 python3 train_demo.py --embedding_type glove --lr 1e-1
```
**Bert Embedding Commain**
```bash
 python3 train_demo.py --embedding_type bert --lr 1e-2 --max_length 60
```
**Result**

|embedding_type| learn_rate | pl_weight| train_iter |  ACC    | 
|:------------:|:---------: | :-------:| :---------:| :------:| 
| glove        |    1e-1    |   1e-1   |    25000   |  78.23  |
| bert         |    1e-2    |   1e-1   |    8000    |  88.45  |


### Incremental Model Training 
Open the directory "IncreProtoNet/IncrementalFewShotModel", and run the python file "train_demo.py". 
The saved model are in the directory "IncreProtoNet/IncrementalFewShotModel/checkpoint/". 
The specific commands and several important hyper-parameters are as follows:

**Glove Embedding Command**
#### 1-shot 
```bash
python3.6 train_demo.py --embedding_type glove --pl_weight 0.1 \
--base_lr 0.1 --train_baseN 30 --triplet_num 10 --margin 5.0 \
--learn_rate 1e-1 --K 1 
```
#### 5-shot
```bash
python3.6 train_demo.py --embedding_type glove --pl_weight 0.1 \
--base_lr 0.1 --train_baseN 30 --triplet_num 10 --margin 5.0 \
--learn_rate 1e-1 --K 5
```
|   Setting   |   Base   |   Novel  |   Both   |
|:-----------:| :-------:| :-------:| :-------:|
|    1-shot   |   70.96  |   48.38  |   69.36  |
|    5-shot   |   72.54  |   61.57  |   71.54  |

**BERT Embedding Command**
#### 1-shot 
```bash
python3 train_demo.py --embedding_type bert \
--pl_weight 0.1 --base_lr 0.01 --train_baseN 30 --triplet_num 10 --margin 10.0 \
--learn_rate 1e-3 --K 1 --val_step 100 --val_iter 100 --max_length 60
```
#### 5-shot
```bash
CUDA_VISIBLE_DEVICES=0,1,3 python3 train_demo.py --embedding_type bert \
--pl_weight 0.1 --base_lr 0.01 --train_baseN 30 --triplet_num 10 \
--margin 10.0 --learn_rate 2e-3 --K 5 --val_step 100 --val_iter 100 \
--max_length 60 --test_Q 1
```
|   Setting   |   Base   |   Novel  |   Both   |
|:-----------:| :-------:| :-------:| :-------:|
|    1-shot   |   82.10  |   60.15  |   80.65  |
|    5-shot   |   84.64  |   65.77  |   82.26  |
