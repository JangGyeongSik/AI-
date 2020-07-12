#data 958 9,1
#학습용 9 테스트용 1
#O->1 X->-1  
#positive 1 negative 0 blank 0

import tensorflow as tf
import numpy as np
from numpy import genfromtxt
import sklearn.model_selection
import random
data = genfromtxt('uci-dataset.csv', delimiter=',')

#print(test, end=" ") 값이 제대로 불러지는지 print
NUM_ITER= 10000
learning_rate= 0.01

np.random.shuffle(data)
x_data = np.array(data[:700, 0:-1], np.float32) #700개의 train_data
y_data = np.array(data[:700, [-1]], np.float32) #700개의 train_data positive , negative 값 
y = np.reshape(y_data, [700,1])  


X = tf.placeholder(tf.float32, shape=[None,9])
Y = tf.placeholder(tf.float32, shape=[None,1])
NUM_HIDDEN=27
#train data

x_test_data=np.array(data[300:-1, 0:-1], np.float32) #x_test_data 300개
y_test_data=np.array(data[300:-1, [-1]], np.float32) #y_test_data 300개 

W = tf.Variable(tf.random_uniform([9, NUM_HIDDEN], -0.5, 0.5))
B = tf.Variable(tf.random_uniform([NUM_HIDDEN], -0.5, 0.5))
hiddenLayer = tf.add(tf.matmul(X, W),B)

W2 = tf.Variable(tf.random_normal([NUM_HIDDEN,1], -0.5, 0.5))
B2 = tf.Variable(tf.random_normal([1], -0.5, 0.5))

yHat = tf.add(tf.matmul(hiddenLayer, W2), B2)

cost = tf.reduce_mean(tf.square(Y - yHat)) #기회비용 함수 
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)# 경사하강법
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

predicted = tf.cast(yHat>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype =tf.float32)) #일괄처리값 계산할때 유용

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #변수 초기화 

error_list = []
w = sess.run(W)
b = sess.run(B)

for k in range(NUM_ITER): #학습 
    e, _= sess.run([cost, optimizer], feed_dict={X:x_data, Y:y})
    error_list.append(e)
    w = sess.run(W).tolist()
    b = sess.run(B).tolist() [0]
    #print(k+1, w, b) #학습시키는 과정 


  
w = sess.run(W)
b = sess.run(B)
print("w=", w)
print("b=", b)
yhat, p, a = sess.run([yHat, predicted, accuracy], feed_dict={X:x_data, Y:y}) #학습 데이터
print("학습률(train) = ", a)

pred=sess.run(predicted, feed_dict={X:x_test_data})
acouracy2 = sess.run(accuracy, feed_dict={X:x_test_data, Y:y_test_data}) # 테스트 데이터 
print("학습률(test) = ", acouracy2)
print("에러율(error_list)= ", error_list[len(error_list)-1]) #에러율 출력 
  
