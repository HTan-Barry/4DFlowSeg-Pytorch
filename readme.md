# 4DFlowSeg

This work is modified by 4DFlowNet project created by: Edward et al. (2020). [GitHub](https://github.com/EdwardFerdian/4DFlowNet), [Paper](https://doi.org/10.3389/fphy.2020.00138)

## 1. Installation

The code has been tested in:
>Python 3.6
>
>Pytorch 1.7

A virtual environment is recommend to created.

### 1.1 PIP

Firstly, install Virtualenv 

```bash
pip install virtualenv
```

Then, create a virtual env

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### 1.2 Conda
If you already install conda:

```bash
conda env create -f environment.yaml
```

## 2. Dataset
To prepare training or validation dataset, we assume a High resolution CFD dataset is available. As an example we have provided this under ```/data/example_data_HR.h5```

### 2.1 Create lower Dataset

```
python prepare_lowres_dataset.py
```

### 2.2 Create batch data

```
bash scripts/create_dataset.sh
```

The parameters in create_dataset.sh:

>dataset: type od dataset (training, validating, testing)
>
>mask_list: all of the setting of the mask threshold

## 3. Training session

```
bash scripts/train.sh

bash scripts/train_seg.sh
```

## 4. Testing session and inference

```
bash scripts/inference.sh
```