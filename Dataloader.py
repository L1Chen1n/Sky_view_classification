import os
import pandas as pd

# Create a data frame to store the dataset and use number to represent a class
def initImg(root, matrix_label):
    label_lst = []
    class_lst = []
    img_lst = []
    label = 0
    for file in os.listdir(root):
        class_path = os.path.join(root, file)
        for fn in os.listdir(class_path):
            img_path = os.path.join(class_path, fn)
            img_lst.append(img_path)
            class_lst.append(file)
            if file not in matrix_label:
                matrix_label.append(file)
            label_lst.append(label)
        label += 1

    return pd.DataFrame({
        "filename": img_lst,
        "class": class_lst,
        "label": label_lst
    })
