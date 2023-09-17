from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import argparse   

def create_confusion_matrix(filename):
    # Data parsing
    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        row = line.strip("\n").split(", ")
        data.append(row)
    data = np.transpose(data)

    # F1-Score
    f1 = metrics.f1_score(data[0],data[1], average="weighted")
    print(f1)
    
    #Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(data[0], data[1])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Neutral", "Sad", "Fear", "Happy"])
    cm_display.plot()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to the data")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    create_confusion_matrix(args.file)
