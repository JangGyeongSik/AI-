'''
ref:https://nasirml.wordpress.com/2017/11/19/single-layer-perceptron-in-tensorflow/
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

NUM_ITER = 1000
learning_rate = 0.01


x = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) # 4x2,input
y = np.array([0, 0, 1, 0], np.float32) #  AND
#y = np.array([0, 1, 1, 1], np.float32) # OR 
y = np.reshape(y, [4,1]) # convert to 4x1

#tensorflow
X = tf.placeholder(tf.float32, shape=[4, 2])
Y = tf.placeholder(tf.float32, shape=[4, 1])
 
##W = tf.Variable(tf.zeros([2, 1]), tf.float32)
##B = tf.Variable(tf.zeros([1, 1]), tf.float32)
W = tf.Variable(tf.random_uniform([2, 1], -0.5, 0.5))
B = tf.Variable(tf.random_uniform([1, 1], -0.5, 0.5))

yHat = tf.sigmoid( tf.add(tf.matmul(X, W), B) ) # 4x1
err = Y - yHat

deltaW = tf.matmul(tf.transpose(X), err ) # 2x1
deltaB = tf.reduce_sum(err, 0) # 1x1

W_ = W + learning_rate * deltaW
B_ = B + learning_rate * deltaB
step = tf.group(W.assign(W_), B.assign(B_)) #update
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
for k in range(NUM_ITER):
    sess.run([step], feed_dict={X: x, Y: y})
w = sess.run(W)
b = sess.run(B)
sess.close()
print('w=', w)
print('b=', b)

## matplotlib
plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1]+0.2)])
plot_y = - 1 / w[1] * (w[0] * plot_x + b)
plot_y = np.reshape(plot_y, [2, -1])

plt.scatter(x[:, 0], x[:, 1], c=y.flatten(), s=100, cmap='viridis')
for i, sample in enumerate(x):
    if y[i] <= 0:
        plt.scatter(sample[0],sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0],sample[1], s=120, marker='+', linewidths=2)

plt.plot(plot_x, plot_y, color='k', linewidth=2)
plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25])
##plt.axes().set_aspect('equal')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

