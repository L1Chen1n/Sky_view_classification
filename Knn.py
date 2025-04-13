from sklearn.neighbors import KNeighborsClassifier
from Dataloader import initImg, dataloader
from Eval_methods import eval
from sklearn.preprocessing import StandardScaler

#dataset root path
root_dir = 'Aerial_Landscapes'

matrix_lst = []
df = initImg(root_dir, matrix_lst)

X_train, y_train, X_test, y_test = dataloader(df, 'sift')

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("training complete")

# Predict and evaluate
y_pred = knn.predict(X_test)

eval(y_test, y_pred, matrix_lst)