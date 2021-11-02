## MNIST - Classification
An ML / MLOps classification project on MNIST - Database of Handwritten Digits implemented using convolutional neural network (CNN) with the help of PyTorch and DVC libraries.

DAGsHub Link (for experimentation and pipelining): [Click Here!](https://dagshub.com/swarajpande4/Digit-Classification)

<!-- 
<br>

### To build from source

 -->

<br>

### Files and Directory Structure
    .
    ├── .github/workflows
        └── cml.yaml                    // CML workflow for GitHub Actions
    ├── code
        ├── eval.py                     // Evaluation metrics script
        ├── featurization.py            // Featurization script
        ├── get_data.py                 // Fetches the datasets for CML container (GitHub Actions Job)
        ├── model_class.py              // Model Class script
        └── train_model.py              // Trains the model instance
    ├── data
        ├── model.pkl                   (DVC)
        ├── norms_params.json           (DVC)
        ├── processed_test_data.npy     (DVC)
        ├── processed_train_data.npy    (DVC)
        ├── test_data.csv               (DVC)
        ├── train_data.csv              (DVC)
        ├── train_data.csv.dvc
        └── test_data.csv.dvc        
    ├── metrics 
        ├── confmat.png                 // Confusion matrix displayed by GitHub Actions
        ├── eval.json                   // Evaluation metrics for pipeline 
        ├── metrics.txt                 // Evaluation metrics displayed by Github Actions                       
        └── train_metric.json           
    ├── notebook
        └── notebook.ipynb              // Jupyter Notebook 
    ├── dvc.lock
    ├── dvc.yaml
    ├── requirements.txt
    └── README.md

<br>