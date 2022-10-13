# 월간 데이콘 코드 유사성 판단 AI 경진대회

### Public score 4th 0.97989 | Private score 5th 0.97848

* 주최 : DACON
* 주관 : DACON
* [https://dacon.io/competitions/official/235900/overview/description](https://dacon.io/competitions/official/235900/overview/description)


### Overview
    - Negative pair를 bm25 알고리즘을 이용하여 각 code마다 k(=5)개의 다른 알고리즘을 코드를 추출하여 데이터셋을 구성.
    - CodeBert-small 백본 기반의 Cross Encoder를 이용하여 5-fold로 학습
    - BinaryClassificationEvaluator를 이용하여 threshold 조정을 통한 성능 향상

## Usage
- `preprocessing.py`를 통해 5-fold train/val.csv 생성

1. Installation
    ```bash
    conda create -n <your_env> python=3.8
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
    #Or you can install a suitable Pytorch version for you from https://pytorch.org/get-started/locally/
    
    pip install -r requirements.txt
    ```
    
2. Download data.zip from[https://dacon.io/competitions/official/235900/data](https://dacon.io/competitions/official/235900/data) to data path.
    ```bash
    #./workspace
    mkdir data
    cd data #And Download data to ./workspace/data/
 
    unzip open.zip
    ```
    
3. Preprocessing
    ```bash
    python preprocessing.py
    ```
    
4. Training
   ```bash
   sh train.sh
   ```
   
5. Test
   ```bash
   sh test.sh
   ```

### Directory Structure
```
/workspace
├── data/
│   ├── code/
│   │    ├── problem001/
│   │    │        ├── problem001_1.py
│   │    │        ├── ...
│   │    └── problem300/
│   │
│   ├── test.csv
│   ├── sample_submission.csv         
│   ├── train_<fold>.csv
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── test.py
│   ├── train.sh
│   ├── test.sh

```


