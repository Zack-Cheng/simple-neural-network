import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

def load_data_sets(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    data = []
    for line in lines:
        x = float(line.split(' ')[0])
        y = float(line.split(' ')[1])
        z = int(line.split(' ')[2])
        data.append([x, y, z])
    return np.array(data)

def model_train(model, data, epoch, batch_size,  grad_check):
    loss= []
    with trange(epoch) as tr:
        for _ in tr:
            idx = np.random.choice(len(data), batch_size)
            item = data[idx]
            x = item[:, :2].T
            t = item[:, 2]
            params = model.gradient(x, t)
            loss.append(params['loss'])
            if grad_check:
                num_grad_params = model.numerical_gradient(x, t)
                diff = model.gradient_check(params, num_grad_params)
                tr.set_postfix(
                    diff_gdw = 'normal' if diff['dw'] < 1e-7 else 'excess',
                    diff_gdb = 'normal' if diff['db'] < 1e-7 else 'excess'
                )
            model.update(params)
    return loss

def plot_model(model, data):
    n = 500
    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    y_min = np.min(data[:, 1])
    y_max = np.max(data[:, 1])

    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xv, yv = np.meshgrid(x, y)
    z = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            item = np.array([[xv[i, j]], [yv[i, j]]])
            z[i, j] = 0 if model.predict(item).item() < 0.5 else 1

    X = [ item[0] for item in data ]
    Y = [ item[1] for item in data ]
    L = [ item[2] for item in data ]

    plt.figure('Model Visualization')
    plt.contourf(xv, yv, z, cmap=plt.cm.Spectral)
    plt.scatter(X, Y, c=L, edgecolors='k', cmap=plt.cm.Spectral)
    plt.show()

def numerical_gradient(f, x):
    h = 1e-5
    gradient = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        v1 = f(x)
        x[idx] = float(tmp_val) - h
        v2 = f(x)

        gradient[idx] = (v1 - v2) / (h*2)
        x[idx] = tmp_val

        it.iternext()
    return gradient
