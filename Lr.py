from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from Dataloader import initImg, dataloader
from Eval_methods import eval

#dataset root path
root_dir = 'Aerial_Landscapes'
max_iter = 800

matrix_lst = []
df = initImg(root_dir, matrix_lst)

X_train, y_train, X_test, y_test = dataloader(df, 'lbp')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(solver='lbfgs', max_iter=max_iter)
lr.fit(X_train_scaled, y_train)
print("Training complete !")

y_pred = lr.predict(X_test_scaled)

eval(y_test, y_pred, matrix_lst)