import numpy as np
import matplotlib.pyplot as plt
from lib.Utils import load_data_sets, model_train, plot_model
from lib.Model import Model
from lib.Layers import Affine, Sigmoid, Cost, Tanh


## PARAMATERS ##
FILE_PATH = './patterns/pattern1.pat'
TRAIN_EPOCH = 100000
LEARNING_RATE = 0.065
HIDDEN_UNIT = 5
BATCH_SIZE = 100
GRAD_CHK = False
################


# Load data sets
data = load_data_sets(FILE_PATH)

# Create layers
params = {}
params['init_scale'] = 0.001
params['learning_rate'] = LEARNING_RATE
params['w_h'] = {'affine1': 2, 'affine2': HIDDEN_UNIT}
params['w_w'] = {'affine1': HIDDEN_UNIT, 'affine2': 1}

affine1 = Affine('affine1', params)
affine2 = Affine('affine2', params)
tanh = Tanh()
sigmoid = Sigmoid()
cost_layer = Cost()

# Create Network
layers = [affine1, tanh, affine2, sigmoid]
two_layers_net = Model(layers=layers, cost_layer=cost_layer)

# Training
loss = model_train(
    model=two_layers_net,
    data=data,
    epoch=TRAIN_EPOCH,
    batch_size=BATCH_SIZE,
    grad_check=GRAD_CHK
)

## Plot loss chart
plt.figure('Loss')
plt.plot(loss)

# Plot trained model
plot_model(model=two_layers_net, data=data)
