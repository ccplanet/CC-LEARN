import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os
# 显存控制
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    # [0, 255] -> [-1,1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255 - 1.
    y = tf.cast(y, dtype=tf.int32)

    return x, y


batchsz = 128
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
# 挤压维度 [10k,1]->[10k]
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
print(x.shape, y.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)

# 检查一个样本
db_iter = iter(train_db)
sample = next(db_iter)
print(sample[0].shape, sample[1].shape)


# 自定义层
class MyDense(layers.Layer):
    # 代替 layers.Dense()
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        # self.bias = self.add_variable('b',[outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x


# 自定义网络
class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # self.fc1 = MyDense(32 * 32 * 3, 256)
        # self.fc2 = MyDense(256, 128)
        # self.fc3 = MyDense(128, 64)
        # self.fc4 = MyDense(64, 32)
        # self.fc5 = MyDense(32, 10)
        self.fc1 = MyDense(32 * 32 * 3, 256)
        self.fc2 = MyDense(256, 256)
        self.fc3 = MyDense(256, 256)
        self.fc4 = MyDense(256, 256)
        self.fc5 = MyDense(256, 10)


    def call(self, inputs, training=None):
        # inputs: [b, 32, 32, 3
        x = tf.reshape(inputs, [-1, 32 * 32 * 3])
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=5, validation_data=test_db, validation_freq=1)

# 模型保存
network.evaluate(test_db)
network.save_weights('ckpt/weights.ckpt')
del network
print('save to ckpt/weights.ckpt')

# 加载测试
network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print('loaded weight')
network.evaluate(test_db)