import math
import numpy as np
import matplotlib.pyplot as plt

from micrograd_engine import MLP, ActivationLayer, Value, MSELoss, Optimizer

mlp = MLP()
mlp.add_layer(1, 5)
mlp.add_activation_layer(ActivationLayer("relu"))
mlp.add_layer(5, 5)
mlp.add_activation_layer(ActivationLayer("relu"))
mlp.add_layer(5, 1)


# print(mlp)

X = np.linspace(0, 5, 50)
y = np.sin(X)

# plt.plot(X, y)
# plt.show()

X = [Value(i) for i in X]
y = [Value(i) for i in y]

loss_fn = MSELoss
optimizer = Optimizer(mlp, 0.01)

epochs = 50
epoch_vals = list(range(1, epochs+1))
loss_vals = []

for epoch in range(epochs):
    y_pred = [mlp([i]) for i in X]
    y_pred = [i[0] for i in y_pred]

    loss = loss_fn(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_vals.append(loss.data)

plt.plot(epoch_vals, loss_vals)
plt.show()
