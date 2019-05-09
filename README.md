# Ensemble Deep Learning Ransomware Detector
A Deep Learning ensemble that classifies Windows executable files as either benign, ransomware, or other malware.
This program was developed as part of my dissertation for my BSc (Hons) Computer Science course at the University of Lincoln: 'Ransomware Detection Using Deep Learning Ensemble' in which it is demonstrated to achieve 96% accuracy in classifying a test set of 3000 PE files not seen in the model's training.

# Setup
For the GUI detector program `ensemblePredict.py`, the following python packages must be installed: tensorflow, keras, h5py, capstone, pefile, numpy, and scikit-learn. These can be installed via the terminal or command prompt command `pip install tensorflow keras h5py capstone pefile numpy scikit-learn`. Then simpy run the script with `python ensemblePredict.py`. You should be greeted by a file selection dialog with which you can select one or more `.exe` files, then click `Open` and the deep learning ensemble will predict if they are benign, ransomware, or other malware.

Source code for traning the models, pre-processing samples, and gathering samples via data mining will be released shortly.
