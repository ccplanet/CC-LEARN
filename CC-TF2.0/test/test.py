import tensorflow as tf

print(tf.__version__)

print(tf.test.is_gpu_available())

# 1.创建输入张量
a = tf.constant(2.)
b = tf.constant(4.)
# 2.直接计算并打印
print('a*b=', a*b)
