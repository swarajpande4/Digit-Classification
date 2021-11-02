"""
Evaluate model performance
"""
import pickle
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import torch

def eval_model():
    # Loading test data
    print("Loading data and model...")
    test_data = np.load('./data/processed_test_data.npy')

    # Loading trained model
    with open('./data/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Switching model to evaluation (inference) mode
    model.eval()

    print("done.")

    # Dividing loaded data-set into data and labels
    labels = test_data[:, 0]
    data = torch.Tensor(test_data[:, 1:].reshape([test_data.shape[0], 1, 28, 28]))

    # Running model on test data
    print("Running model on test data...")
    predictions = model(data).max(1, keepdim=True)[1].cpu().data.numpy()
    print("done.")

    # Calculating metric scores
    print("Calculating metrics...")
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')

    metrics = {'accuracy': accuracy, 'f1_score': f1, 'recall': recall}

    conf_mat = confusion_matrix(labels, predictions)
    conf_mat_cnn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    f,ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat_cnn, annot=True, linewidths=0.01,cmap="cubehelix",linecolor="gray", fmt= '.1f',ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    # Saving metrics for cml
    plt.savefig('metrics/confmat.png', dpi=120)
    with open("metrics/metrics.txt", 'w') as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"f1 Score: {f1}\n")
        file.write(f"Recall Score: {recall}\n")

    # Saving metrics to json file
    json_object = json.dumps(metrics, indent=4)
    with open('./metrics/eval.json', 'w') as f:
        f.write(json_object)
    print("done.")


if __name__ == '__main__':
    eval_model()