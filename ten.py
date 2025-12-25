import tensorflow as tf
import numpy as np
x = np.array([2020, 2021, 2022, 2023], dtype=float)
y = np.array([2134, 3245, 1245, 2122], dtype=float)
x = x / 2020
y = y / 1000
model = tf.keras.Sequential([tf.keras.Input(shape=(1,)),tf.keras.layers.Dense(1)])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(x, y, epochs=500, verbose=0)
na = model.predict(np.array([[2024 / 2020]], dtype=float))[0]
print(na * 1000)
