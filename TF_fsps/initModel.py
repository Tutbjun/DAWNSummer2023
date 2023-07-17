import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(12, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(12, activation='sigmoid')
])
model.compile(optimizer
    =tf.keras.optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['acc'])

#save model
model.save("model_0.h5")