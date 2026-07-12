# Environment Setup Guide

## 1. Requirements

Recommended:

* Python 3.10+
* NVIDIA GPU (optional)
* CUDA compatible environment

## 2. Create Virtual Environment

Using conda:

```bash
conda create -n emotion_ai python=3.10

conda activate emotion_ai
```

## 3. Install Basic Packages

```bash
pip install numpy
pip install opencv-python
pip install matplotlib
pip install tqdm
```

## 4. Install Deep Learning Framework

GPU version:

```bash
pip install torch torchvision torchaudio
```

Check installation:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

Expected:

```
True
```

## 5. Install Face Detection Tools

```bash
pip install mediapipe
```

## 6. Install Machine Learning Tools

```bash
pip install scikit-learn
pip install pandas
```

## 7. Install Development Tools

```bash
pip install jupyter
pip install ipykernel
```

## 8. Generate requirements.txt

```bash
pip freeze > requirements.txt
```

## Environment Test

Run:

```bash
python test_environment.py
```

Expected:

```
Environment Ready!
```
