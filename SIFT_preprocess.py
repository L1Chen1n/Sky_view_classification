import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split


def preprocessImg(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    sharpened = cv2.addWeighted(img, 1, laplacian, 1.0, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(sharpened)
    return clahe_img


def build_bow_histogram(descriptors, kmeans, n_clusters):
    if descriptors is None:
        return np.zeros(n_clusters)

    descriptors = np.asarray(descriptors, dtype=np.float32)
    words = kmeans.predict(descriptors)

    histogram, _ = np.histogram(words, bins=np.arange(n_clusters + 1))

    return histogram


def get_descriptors(img_path, preprocess=True):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if preprocess:
        gray = preprocessImg(gray)

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(gray, None)

    return descriptors


def descriptorlst(train_df):
    descriptor_lst = []

    for _, row in train_df.iterrows():
        descriptors = get_descriptors(row['filename'], preprocess=True)
        if descriptors is not None:
            descriptor_lst.extend(descriptors)
    descriptor_lst = np.asarray(descriptor_lst, dtype=np.float32)

    return descriptor_lst


def dataloader(data):
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )
    descriptor_lst = descriptorlst(train_df)
    kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
    kmeans.fit(descriptor_lst)
    data_type = ['train_df', 'test_df']
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for ele in data_type:
        if ele == 'train_df':
            data = train_df
        elif ele == 'test_df':
            data = test_df

        for _, row in data.iterrows():
            descriptors = get_descriptors(row['filename'], preprocess=False)
            hist = build_bow_histogram(descriptors, kmeans, 100)
            if ele == 'train_df':
                X_train.append(hist)
                y_train.append(row['label'])
            else:
                X_test.append(hist)
                y_test.append(row['label'])

    return X_train, y_train, X_test, y_test