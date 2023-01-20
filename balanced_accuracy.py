import numpy as np

def balanced_accuracy(p_test, y_test):
    print(len(p_test))
    print(len(y_test))
    accs = []
    for cls in range(10):
        mask = (y_test == cls)
        cls_acc = (p_test == cls)[mask].mean() # Accuracy for rows of class cls
        accs.append(cls_acc)
    return np.mean(accs) # Final balanced accuracy
    