import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# Enable profiler
tf.profiler.experimental.start('/root/home-nvidia/uqo-masters-thesis/logs')

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights=None)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

tf.profiler.experimental.stop()

