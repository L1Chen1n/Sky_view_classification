import os
import pandas as pd
from sklearn.model_selection import train_test_split
from SIFT_preprocess import build_bow_histogram, get_descriptors, descriptorlst
from sklearn.cluster import MiniBatchKMeans
from LBP_preprocess import extract_lbp_features
import numpy as np

number_of_data = {'Forest': 550, 'Lake': 400, 'Mountain': 350, 'Port': 250, 'Railway': 200, 'Parking': 300, 
                  'City': 650, 'Highway': 450, 'Residential': 150, 'Agriculture': 700, 'Grassland': 500, 
                  'Beach': 700, 'Desert': 600, 'Airport': 700, 'River': 100}

def imBalanced_data(root, matrix_label):
    training_label_lst, training_class_lst, training_img_lst = [], [], []
    testing_label_lst, testing_class_lst, testing_img_lst = [], [], []
    label = 0
    for file in os.listdir(root):
        class_path = os.path.join(root, file)
        count = 1
        for fn in os.listdir(class_path):
            if count <= number_of_data[file]:
                img_path = os.path.join(class_path, fn)
                training_img_lst.append(img_path)
                training_class_lst.append(file)
                if file not in matrix_label:
                    matrix_label.append(file)
                training_label_lst.append(label)
                count += 1
            else:
                img_path = os.path.join(class_path, fn)
                testing_img_lst.append(img_path)
                testing_class_lst.append(file)
                testing_label_lst.append(label)
        label += 1
    # print(len(training_img_lst),len(training_class_lst),len(training_label_lst),len(),len(),len())
    return pd.DataFrame({
                "filename": training_img_lst,
                "class": training_class_lst,
                "label": training_label_lst
            }),pd.DataFrame({
                            "filename": testing_img_lst,
                            "class": testing_class_lst,
                            "label": testing_label_lst
                        })

def imBalanced_dataloader(training_data, testing_data, preprocess):
    train_df, test_df = training_data, testing_data

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

