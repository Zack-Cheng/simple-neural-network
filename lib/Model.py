import numpy as np
from . import Utils
from . import Layers

class Model(object):
    def __init__(self, layers, cost_layer):
        self.layers = layers
        self.cost_layer = cost_layer

    def predict(self, x):
        out = None
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = layer.forward(x)
            else:
                out = layer.forward(out)
        return out

    def gradient(self, x, t):
        y = self.predict(x)
        loss = self.cost_layer.forward(t, y)
        dout = self.cost_layer.backward()
        rst = {
            'loss': None,
            'grad': {}
        }
        rst['loss'] = (np.sum(loss, axis=1) / loss.shape[1]).item()
        for layer in reversed(self.layers):
            if isinstance(layer, Layers.Affine):
                dout, dw, db = layer.backward(dout)
                layer_name = layer.name
                rst['grad'][layer_name] = {}
                rst['grad'][layer_name]['dw'] = dw
                rst['grad'][layer_name]['db'] = db
            else:
                dout = layer.backward(dout)
        return rst

    def numerical_gradient(self, x, t):
        def _loss(_):
            y = self.predict(x)
            loss = np.sum(self.cost_layer.forward(t, y), axis=1) / y.shape[1]
            return loss
        rst = {
            'loss': None,
            'grad': {}
        }
        rst['loss'] = _loss(None)
        for layer in self.layers:
            if not isinstance(layer, Layers.Affine):
                continue
            dw = Utils.numerical_gradient(_loss, layer.w)
            db = Utils.numerical_gradient(_loss, layer.b)
            layer_name = layer.name
            rst['grad'][layer_name] = {}
            rst['grad'][layer_name]['dw'] = dw
            rst['grad'][layer_name]['db'] = db
        return rst

    def update(self, params):
        for layer_name, grad in params['grad'].items():
            for layer in self.layers:
                if not isinstance(layer, Layers.Affine):
                    continue
                if layer.name == layer_name:
                    layer.update(grad['dw'], grad['db'])

    def gradient_check(self, grad_result, num_grad_result):
        for layer_name, grad in grad_result['grad'].items():
            diff = {}
            for vec, val in grad.items():
                diff[vec] = np.sum(val - num_grad_result['grad'][layer_name][vec])
        return diff

