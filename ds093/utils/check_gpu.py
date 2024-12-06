import tensorflow as tf
from tensorflow.python.client import device_lib


print("build with cuda: ", tf.test.is_built_with_cuda())
print("gpu list: ", tf.config.list_physical_devices("GPU"))
print("list devices: ", device_lib.list_local_devices())
