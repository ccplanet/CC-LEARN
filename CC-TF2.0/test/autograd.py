# 自动求导测试

import tensorflow as tf

# 创建4个张量
x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)


with tf.GradientTape() as tape:# 构建梯度环境
	tape.watch([a, b, c]) # 将w加入梯度跟踪列表
	# 构建计算过程
	y = a**2 * x + b * x + c
# 求导
[dy_da, dy_db, dy_dc] = tape.gradient(y, [a, b, c])
print(dy_da, dy_db, dy_dc)

