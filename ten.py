import tensorflow as tf
import numpy as np

# Data
x = np.array([2020, 2021, 2022, 2023], dtype=float)
y = np.array([2134, 3245, 1245, 2122], dtype=float)

# Scaling
x = x / 2020
y = y / 1000

# Model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Train
model.fit(x, y, epochs=500, verbose=0)

# Predict (NUMPY ARRAY!)
na = model.predict(np.array([[2024 / 2020]], dtype=float))[0]

# Back to original scale
print(na * 1000)
