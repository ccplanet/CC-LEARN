import tensorflow as tf
import torch
print(tf.__version__)

print(tf.test.is_gpu_available())
# 1.创建输入张量
a = tf.constant(2.)
b = tf.constant(4.)
# 2.手写数字识别初试.直接计算并打印
print('a*b=', a*b)

print(torch.cuda.is_available())
a = torch.constant(2.)
b = torch.constant(4.)
# 2.手写数字识别初试.直接计算并打印
c = add(a,b)
print('a*b=', )