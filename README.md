# BSP5

### Requirements
To install the necessary libraries, use on Windows:
```
pip install -r requirements.txt
```
or on Mac/Linux:
```
pip3 install -r requirements.txt
```

### Malware Analysis
Contains the jupyter notebooks for the analysis of the classifiers and the gradient descent evasion attacks.

### Scripts
Execute **evade_pdf** to modify a malicious PDF file.
When the script is running, use:
- **train** to train classifier on the PDF malware dataset
- **evade** to modify a malicious PDF file such that it bypasses the trained classifier
- **quit** to stop the program
