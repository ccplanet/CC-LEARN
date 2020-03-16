import numpy as np


# data = []
# for i in range(100):
# 	x = np.random.uniform(3., 12.)
# 	# mean=0, std=0.1
# 	eps = np.random.normal(0., 0.1)
# 	y = 1.477 * x + 0.089 + eps
# 	data.append([x, y])
# data = np.array(data)
# print(data.shape, data)

# y = wx + b
# 1. 计算损失函数
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # computer mean-squared-error
        totalError += (y - (w * x + b)) ** 2
    # average loss for each point
    return totalError / float(len(points))


# 2.手写数字识别初试. 计算梯度，更新参数
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2.手写数字识别初试(wx+b-y)
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        # grad_w = 2.手写数字识别初试(wx+b-y)*x
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)
    # update w'
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


# 3. 循环更新，梯度下降算法运行主体
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    # 初始化数据
    points = np.genfromtxt("data.csv", delimiter=",")
    # 初始化参数
    learning_rate = 0.0001
    initial_b = 0  # initial y-intercept guess
    initial_w = 0  # initial slope guess
    num_iterations = 1000
    # 计算初始误差
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points))
          )
    print("Running...")
    # 开始训练
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    # 计算完成后误差
    print("After {0} iterations b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()
