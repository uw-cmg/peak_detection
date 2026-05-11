# peak_detection
For APT spectrum peak ranging and identification

Module RangingNN contains a supervised YOLO-based model for ranging the APT M/C specturm. It was trained on expert labeled datasets. 

Module Ionclassifier contains a supervised Recurrent CNN model for identify the ion species of peaks. It was trained on synthetic datasets. 

### Installation 

---
At the in-development stage, please install the package from the github source code:

pip install git+https://github.com/wdwzyyg/peak_detection.git

Python version 3.10.0 is recommended. Older python version may not support the pytorch version used here, and newer version has not been tested. 
You can create an independent python environment by running the command:
```
conda create -n name_of_environment python=3.10.0
conda activate name_of_environment
pip install git+https://github.com/wdwzyyg/peak_detection.git
...
```

### Usage 

---
#### Using the RangingNN and IonClassifier models

Use ML models to predict APT peak ranges and ion types:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor.ipynb)

Dev notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2025-10-14.ipynb)

Dev notebook 11/5/25 (example of using per-spectrum RF):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2025_11_5.ipynb)

Dev notebook 11/18/25 (example of using updated ranging model with iterative peak finding):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2025_11_18.ipynb)

Dev notebook 3/16/26 (example of using new CLI-based code detect_peaks.py):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/APT_Predictor_dev_2026_03_16.ipynb)

Ranging model data creation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uw-cmg/peak_detection/blob/master/DataCreation.ipynb)
