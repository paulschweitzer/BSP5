import numpy as np
import pandas as pd


def modify_sample(x0, args, model, classifier='linear', lr=1e-2, epsilon=1e-4, iterations=200, d_max=50):
    
    gradient = compute_gradient(x0, args, classifier)

    modified = np.zeros(x0.shape)
    modified = evasion_gradient_descent(
        x0=x0,
        gradient=gradient,
        t=lr,
        epsilon=epsilon,
        max_iter=iterations,
        d_max=d_max
        )

    pred = model.predict([modified])
    print(f"Predicted class: {pred}")

    return modified


def compute_gradient(x0, args, classifier='linear'):
    gradient = 0
    if classifier == 'linear':
        gradient = args['gradient_linear']
    elif classifier == 'rbf':
        gradient = svm_gradient(weights=args['weights_rbf'], feature_vector=x0, support_vectors=args['support_vectors_rbf'], kernel='rbf', gamma=args['gamma'])
    elif classifier == 'mlp':
        gradient = mlp_gradient(x0, args['hidden_weights'], args['hidden_bias'], args['output_weights'], args['output_bias'])
    return gradient


def evasion_gradient_descent(x0, gradient, t, epsilon, max_iter, d_max=10):

    m = 0
    x_m = x0
    for i in range(max_iter):

        m += 1
        x_m = x_m - t * gradient
        # x_m = np.maximum(x_m, 0)
        x_m = project(x_m, x0, d_max)
        if np.linalg.norm(x_m - x0) < epsilon:
            break

    x = x_m
    return x_m


def project(x, x0, d_max):
    dist = np.linalg.norm(x - x0)
    if dist > d_max:
        return x0 + d_max * (x - x0) / dist
    return x


def rbf_kernel(x, x_i, gamma=0.0001):
    return np.exp(- gamma * np.pow(np.linalg.norm(x - x_i), 2))

def poly_kernel(x, x_i, d=3, c=1):
    return np.pow((x @ x_i) + c, d)

def gradient_rbf_kernel(x, x_i, gamma):
    return -2 * gamma * np.exp(- gamma * np.pow(np.linalg.norm(x - x_i), 2)) * (x - x_i)

def gradient_poly_kernel(x, x_i, d, c):
    return d * np.pow(d * (x @ x_i + c), d - 1) * x_i


def svm_gradient(weights, feature_vector=None, support_vectors=None, kernel='linear', gamma=0.001):
    if kernel == 'linear':
        return weights
    elif kernel == 'rbf':
        delta_g = np.zeros(feature_vector.shape)
        kernel_gradient = np.zeros(feature_vector.shape)
        for i in range(support_vectors.shape[0]):
            w_i = weights[i]
            x_i = support_vectors[i]
            kernel_gradient = gradient_rbf_kernel(feature_vector, x_i, gamma)
            delta_g = delta_g + w_i * kernel_gradient
        return delta_g
    
    else:
        return -1


def tanh(z):
    return np.tanh(z)

def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def relu(z):
    return max(0, z)

def tanh_derivative(z):
    return 1 - np.pow(tanh(z), 2)

def sigmoid_derivative(z):
    return 1 - sigmoid(z)

def relu_derivative(z):
    return z>=0


def mlp_discriminant(x, hidden_weights, hidden_bias, output_weights, output_bias):
    g = tanh(output_weights.T @ tanh(hidden_weights.T @ x + hidden_bias) + output_bias)
    return g


def mlp_gradient(x, hidden_weights, hidden_bias, output_weights, output_bias):
    g = mlp_discriminant(x, hidden_weights, hidden_bias, output_weights, output_bias)
    delta_k = tanh(hidden_weights.T @ x + hidden_bias)

    delta_g = np.zeros(x.shape)
    for i in range(x.shape[0]):
        # delta_g[i] = g * (1 - np.pow(g, 1)) * (output_weights.T @ (delta_k * (1 - np.pow(delta_k, 1)) * hidden_weights[i]))
        delta_g[i] = g * (1 - np.pow(g, 2)) * (output_weights.T @ (delta_k * (1 - np.pow(delta_k, 2)) * hidden_weights[i]))

    return delta_g


def rename_columns(df: pd.DataFrame):

    column_mapping = {
        "/Page": "pages",
        "/Encrypt": "isEncrypted",
        "/ObjStm": "ObjStm",
        "/JS": "JS",
        "/JavaScript": "Javascript",
        "/AA": "AA",
        "/OpenAction": "OpenAction",
        "/AcroForm": "Acroform",
        "/JBIG2Decode": "JBIG2Decode",
        "/RichMedia": "RichMedia",
        "/Launch": "launch",
        "/EmbeddedFile": "embedded files",
        "/XFA": "XFA",
        "/Colors > 2^24": "Colors"
    }

    df.rename(columns=column_mapping, inplace=True)
    return df
