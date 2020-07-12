'''
XOR by 2-layer MLP
hidden layer nodes: NUM_HIDDEN
output layer node :  1   

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

NUM_ITER = 1000
learning_rate = 0.1
NUM_HIDDEN = 2

# train data
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32) # 4x2,input
y = np.array([0, 1, 1, 0], np.float32) #  XOR
y = np.reshape(y, [4,1]) # convert to 4x1

#tensorflow
X = tf.placeholder(tf.float32, shape=[None, 2]) #[4, 2]
Y = tf.placeholder(tf.float32, shape=[None, 1]) #[4, 1]

W = tf.Variable(tf.random_uniform([2, NUM_HIDDEN], -0.5, 0.5))
B = tf.Variable(tf.random_uniform([NUM_HIDDEN], -0.5, 0.5))

##hiddenLayer  = tf.nn.relu(tf.matmul(X,W) + B)
hiddenLayer = tf.sigmoid( tf.add(tf.matmul(X, W), B) )

W2  = tf.Variable(tf.random_uniform([NUM_HIDDEN,1], -0.5, 0.5))
B2  = tf.Variable(tf.random_uniform([1], -0.5, 0.5))
outLayer =tf.add(tf.matmul(hiddenLayer, W2), B2)
yHat =tf.sigmoid(outLayer)

cost = tf.reduce_sum(tf.square(Y - yHat))
##cost = -tf.reduce_mean(Y * tf.log(yHat) + (1 - Y) * tf.log(1 - yHat))

##optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

predicted = tf.cast(yHat> 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

error_list = []
for k in range(NUM_ITER):
        e, _ = sess.run([cost, optimizer], feed_dict={X:x, Y:y})
        error_list.append(e)

w = sess.run(W)
b = sess.run(B)
print('w=', w)
print('b=', b)

yhat, p, a = sess.run([yHat, predicted, accuracy], feed_dict={X:x,Y:y})
print("yhat = ", yhat)
print("predicted = ", p)
print("accuracy = ", a)

#### matplotlib
xx, yy = np.meshgrid(np.linspace(0.0,1.0,num=100),
                     np.linspace(0.0,1.0,num=100))
test_data = np.c_[xx.ravel(), yy.ravel()]
z = sess.run(predicted, feed_dict={X:test_data})
##print("predicted in sample= ", z)
sess.close()

plt.scatter(test_data[:, 0], test_data[:, 1], c=z.flatten(), s=1, cmap='viridis')
plt.contour(xx, yy, z.reshape(xx.shape))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.plot(range(NUM_ITER), error_list)
plt.show()
