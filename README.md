# Preparations
## Download Dataset
[Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/)
## Download dependencies
pip install -r requirements.txt

# Preprocessing
run the file `preprocessing1.py`   extract the minimum and maximum values of each pair of subsequences from the distance contour matrix of the ECG signals  

run the file `preprocessing2.py`   extract R-R interval signal, R-peak amplitude signal and heart rate variability (HRV) feature 

#  classification
run the file `./models/model.ipynb`  

