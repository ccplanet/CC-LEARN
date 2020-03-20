import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)

    return x, y


# 下载数据集
# (x, y), (x_test, y_test) = datasets.mnist.load_data()
# (x, y), (x_test, y_test) = datasets.cifar10.load_data()
# (x, y), (x_test, y_test) = datasets.cifar100.load_data()
# (x, y), (x_test, y_test) = datasets.imdb.load_data()
# (x, y), (x_test, y_test) = datasets.reuters.load_data()
# (x, y), (x_test, y_test) = datasets.boston_housing.load_data()
(x, y), (x_test, y_test) = datasets.mnist.load_data()
print(x.shape, y.shape)

batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(batchsz)

# 检查一个样本
db_iter = iter(db)
sample = next(db_iter)
print(sample[0].shape, sample[1].shape)

# 5层的全连接层
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # [b, 784] -> [b, 256]
    layers.Dense(128, activation=tf.nn.relu),  # [b, 256] -> [b, 128]
    layers.Dense(64, activation=tf.nn.relu),  # [b, 128] -> [b, 64]
    layers.Dense(32, activation=tf.nn.relu),  # [b, 64] -> [b, 32]  330 = 32*10 + 10
    layers.Dense(10)  # [b, 32] -> [b, 10]
])

model.build(input_shape=[None, 28 * 28])
model.summary()
# w = w -lr * grad
optimizer = optimizers.Adam(lr=1e-3)


def main():

    for epoch in range(10):

        for step, (x, y) in enumerate(db):

            x = tf.reshape(x, [-1, 28 * 28])

            with tf.GradientTape() as tape:
                # [b, 784] -> [b, 10]
                logit = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logit))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logit, from_logits=True))

            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))

        # test
        total_correct = 0
        total_num = 0
        for x,y in db_test:
            x = tf.reshape(x, [-1, 28*28])
            logit = model(x)
            # log ->pro
            prob = tf.nn.softmax(logit, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct/total_num
        print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
