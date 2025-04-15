import os
import pandas as pd
from SIFT_preprocess import build_bow_histogram, get_descriptors, descriptorlst
from LBP_preprocess import extract_lbp_features
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import numpy as np

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


def dataloader(data, preprocess):
    train_index = np.load('train_indices.npy')
    test_index = np.load('val_indices.npy')
    train_df, test_df = data.iloc[train_index], data.iloc[test_index]
    
    X_train, y_train, X_test, y_test = [], [], [], []
    if preprocess == 'sift':
        descriptor_lst = descriptorlst(train_df)
        kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
        kmeans.fit(descriptor_lst)
        for df, X, y in [(train_df, X_train, y_train), (test_df, X_test, y_test)]:
            for _, row in df.iterrows():
                descriptors = get_descriptors(row['filename'], preprocess=False)
                hist = build_bow_histogram(descriptors, kmeans, 100)
                X.append(hist)
                y.append(row['label'])

    elif preprocess == 'lbp':
         for df, X, y in [(train_df, X_train, y_train), (test_df, X_test, y_test)]:
            for _, row in df.iterrows():
                features = extract_lbp_features(row['filename'])
                X.append(features)
                y.append(row['label'])

    return np.array(X_train), y_train, np.array(X_test), y_test