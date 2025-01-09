import numpy as np
import pandas as pd

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def get_data(path: str):
    data = pd.read_csv(path).dropna()
    return data


def preprocess_data(data: pd.DataFrame):
    pdfid_columns = ['obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref', 'header',
                 'pages', 'isEncrypted', 'ObjStm', 'JS', 'Javascript', 'AA', 'OpenAction',
                 'Acroform', 'JBIG2Decode', 'RichMedia', 'launch', 'embedded files', 'XFA', 'Colors',
                 'Class']

    new_labels = {'Malicious': 1, 'Benign': 0}
    data['Class'] = data['Class'].map(new_labels)


    new_labels = {'1(1)': '1', '2(1)': '2', '3(1)': '3', '29(2)': '29', '34(2)': '34', '2(2)': '2', '>': '0', '53(1)': '53', '5(1)': '5', '12(2)': '2', '53(2)': '53', '-1': '0', 
                '(most': '0', '_Pro_Rodeo_Pix_': '0', "_Pro_Rodeo_Pix_'": '0', 'pdfid.py': '0', 'pdfHeader)': '0', 'bytes[endHeader]': '0', 'list': '0', 'unclear': '0', 'Yes': '1', 'No': '0'}

    for col in data.drop(columns=['Class']).columns:
        data[col] = data[col].replace(new_labels)

    return data[pdfid_columns]


def train_linear(path: str):

    data = get_data(path)
    data = preprocess_data(data)

    class_data = data.drop(columns=['header', 'Class'])
    x_train, x_test, y_train, y_test = train_test_split(class_data, data['Class'], test_size=0.2, random_state=77)

    # Scale features
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    C_values = [1, 0.8, 0.5, 0.3, 0.1]
    scores = []
    
    # Determine best kernel parameters
    print("Choosing kernel parameters...")
    for c in C_values:
        svc = SVC(kernel='linear', C=c)
        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        scores.append(f1)
        # print(f"C = {c}, F1: {f1}")
    
    best_c = C_values[scores.index(max(scores))]
    print(f"-> C = {best_c}")

    print("Train classifier...")
    svc = SVC(kernel='linear', C=best_c)
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)

    svm_accuracy = accuracy_score(y_test, y_pred)
    svm_precision = precision_score(y_test, y_pred)
    svm_recall = recall_score(y_test, y_pred)
    svm_f1 = f1_score(y_test, y_pred)

    print(f"""SVM scores
      accuracy score: {svm_accuracy: .3f}
      precision score: {svm_precision: .3f}
      recall score: {svm_recall: .3f}
      f1 score: {svm_f1: .3f}\n""")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # return parameters of the discriminant
    weights_linear = svc.coef_[0]
    bias_linear = svc.intercept_
    gradient_linear = weights_linear

    return weights_linear, svc, scaler

    
def train_rbf(path: str):

    data = get_data(path)
    data = preprocess_data(data)

    class_data = data.drop(columns=['header', 'Class'])
    x_train, x_test, y_train, y_test = train_test_split(class_data, data['Class'], test_size=0.2, random_state=77)

    # Scale features
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Determine best kernel parameters
    print("Choosing kernel parameters...")
    gamma_values = [10, 5, 1, 1/1000, 1/x_train.shape[0], 1/10**6]
    scores = []

    for g in gamma_values:
        svc_rbf = SVC(kernel='rbf', gamma=g)
        svc_rbf.fit(x_train, y_train)
        y_pred = svc_rbf.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        scores.append(f1)
        # print(f"Gamma = {g}, F1: {f1}")

    best_gamma = gamma_values[scores.index(max(scores))]
    print(f"-> gamma = {best_gamma}")

    print("Train classifier...")
    svc_rbf = SVC(kernel='rbf', gamma=best_gamma)
    svc_rbf.fit(x_train, y_train)

    y_pred = svc_rbf.predict(x_test)

    svm_accuracy = accuracy_score(y_test, y_pred)
    svm_precision = precision_score(y_test, y_pred)
    svm_recall = recall_score(y_test, y_pred)
    svm_f1 = f1_score(y_test, y_pred)

    print(f"""SVM scores
      accuracy score: {svm_accuracy: .3f}
      precision score: {svm_precision: .3f}
      recall score: {svm_recall: .3f}
      f1 score: {svm_f1: .3f}\n""")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # return parameters of the discriminant
    gamma = svc_rbf.get_params()['gamma']
    weights_rbf = svc_rbf.dual_coef_[0]
    support_vectors_rbf = svc_rbf.support_vectors_

    return gamma, weights_rbf, support_vectors_rbf, svc_rbf, scaler


def train_mlp(path: str):

    data = get_data(path)
    data = preprocess_data(data)

    class_data = data.drop(columns=['header', 'Class'])
    x_train_mlp, x_test_mlp, y_train_mlp, y_test_mlp = train_test_split(class_data, data['Class'], test_size=0.2, random_state=77)

    # Normalize features
    normalizer = Normalizer()
    x_train_mlp = normalizer.fit_transform(x_train_mlp)
    x_test_mlp = normalizer.transform(x_test_mlp)
    
    # Train classifier
    print("Train MLP...")
    hidden_layers = (128,)
    activation = 'tanh'
    solver = 'adam'

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver=solver)
    mlp.fit(x_train_mlp, y_train_mlp)

    y_pred = mlp.predict(x_test_mlp)

    mlp_accuracy = accuracy_score(y_test_mlp, y_pred)
    mlp_precision = precision_score(y_test_mlp, y_pred)
    mlp_recall = recall_score(y_test_mlp, y_pred)
    mlp_f1 = f1_score(y_test_mlp, y_pred)

    print(f"""MLP scores
      accuracy score: {mlp_accuracy: .3f}
      precision score: {mlp_precision: .3f}
      recall score: {mlp_recall: .3f}
      f1 score: {mlp_f1: .3f}\n""")
    print(f"Confusion matrix:\n{confusion_matrix(y_test_mlp, y_pred)}")

    weights = mlp.coefs_
    bias = mlp.intercepts_

    return weights, bias, mlp, normalizer
