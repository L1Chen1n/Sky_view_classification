import os
import pandas as pd
from sklearn.model_selection import train_test_split
from SIFT_preprocess import build_bow_histogram, get_descriptors, descriptorlst
from sklearn.cluster import MiniBatchKMeans
from LBP_preprocess import extract_lbp_features
import numpy as np

number_of_data = {'Forest': 300, 'Lake': 50, 'Mountain': 50, 'Port': 50, 'Railway': 50, 'Parking': 50, 
                  'City': 500, 'Highway': 100, 'Residential': 50, 'Agriculture': 800, 'Grassland': 200, 
                  'Beach': 600, 'Desert': 400, 'Airport': 700, 'River': 50}

def imBalanced_data(root, matrix_label):
    label_lst = []
    class_lst = []
    img_lst = []
    label = 0
    for file in os.listdir(root):
        class_path = os.path.join(root, file)
        count = 0
        for fn in os.listdir(class_path):
            img_path = os.path.join(class_path, fn)
            img_lst.append(img_path)
            class_lst.append(file)
            if file not in matrix_label:
                matrix_label.append(file)
            label_lst.append(label)
            count += 1
            if count == number_of_data[file]:
                break
            

        label += 1

    return pd.DataFrame({
        "filename": img_lst,
        "class": class_lst,
        "label": label_lst
    })

def imBalanced_dataloader(data, preprocess):
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )
    
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

