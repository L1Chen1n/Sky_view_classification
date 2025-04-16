from sklearn.neighbors import KNeighborsClassifier
from Dataloader import initImg, dataloader
from Eval_methods import eval
from imblearn.over_sampling import SMOTE
from Dataloader_imbalanced import imBalanced_data, imBalanced_dataloader

#dataset root path
root_dir = 'Aerial_Landscapes'

matrix_lst = []
df = initImg(root_dir, matrix_lst)

# X_train, y_train, X_test, y_test = dataloader(df, 'sift')

training_df, testing_df = imBalanced_data(root_dir, matrix_lst)
# df2 = imBalanced_data(root_dir, matrix_lst)

X_train, y_train, X_test, y_test = imBalanced_dataloader(training_df, testing_df, 'sift')
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled, y_resampled)
print("training complete")

# Predict and evaluate
y_pred = knn.predict(X_test)

eval(y_test, y_pred, matrix_lst)