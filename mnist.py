import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.python.eager import profiler

# Enable profiler
tf.profiler.experimental.start('/root/home-nvidia/uqo-masters-thesis/logs')

print("TF version" + tf.__version__) 

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


# Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)  # Reshape training data
x_test = x_test.reshape(-1, 784)  # Reshape test data
x_train = x_train.astype("float32") / 255.0  # Normalize pixel values
x_test = x_test.astype("float32") / 255.0

# Build and compile the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Create a TensorBoard callback
logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# debug info
# tf.debugging.experimental.enable_dump_debug_info(logs, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)


# tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                #  histogram_freq = 1,
                                                #  profile_batch = '500,520')

# tf.profiler.experimental.start(logdir=logs)

# Train the model
model.fit(x_train, y_train, epochs=1, batch_size=64)
# model.fit(x_train, y_train, epochs=3, batch_size=64, callbacks = [tboard_callback])

tf.profiler.experimental.stop()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

model.save('mnist_model.h5')