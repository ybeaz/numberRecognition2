## Libraries import
import tensorflow as tf

## Test GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
print('')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True