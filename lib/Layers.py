import numpy as np

class Affine(object):
    def __init__(self, name, params):
        init_scale = params['init_scale']
        w_h = params['w_h'][name]
        w_w = params['w_w'][name]
        learning_rate = params['learning_rate']

        self.w = init_scale * np.random.randn(w_h, w_w)
        self.b = np.zeros((w_w, 1))
        self.name = name
        self.x = None
        self.learning_rate = learning_rate

    def forward(self, x):
        self.x = x
        z = np.dot(self.w.transpose(), x) + self.b
        return z

    def backward(self, dout):
        dx = np.dot(self.w, dout)
        train_set_size = self.x.shape[1]
        dw = np.dot(self.x, dout.transpose()) / train_set_size
        db = np.sum(dout, axis=1).reshape(self.b.shape) / train_set_size
        return dx, dw, db

    def update(self, dw, db):
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

class Sigmoid(object):
    def forward(self, x):
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1.0 - self.y)

class Tanh(object):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return (1 - (self.out ** 2)) * dout

class ReLU(object):
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Cost(object):
    def forward(self, t, y):
        self.y = y
        self.t = t
        loss = -(t * np.log(y) + (1.0 - t) * np.log(1.0 - y))
        return loss

    def backward(self):
        dout = -(self.t / self.y) + ((1.0 - self.t) / (1.0 - self.y))
        return dout

