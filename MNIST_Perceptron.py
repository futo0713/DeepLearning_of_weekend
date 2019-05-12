import function
import numpy as np
import matplotlib.pyplot as plt
import time

#minist_load
save_file = '/Users/tsutsumifutoshishi/Desktop/MNIST_test/mnist.pkl'
X_train,T_train,X_test,T_test = function.mnist(save_file)

# function.show(X_train,T_train,0)

#initial setting
W = np.random.randn(784, 10)
B = np.random.randn(1,10)
learning_rate = 0.001

E_save = []
accuracy_save = []
start_time = time.time()

batch_size = 100

#iteration
num_of_itr=3000
for i in range(num_of_itr):

    X_batch,T_batch = function.batch(X_train,T_train,batch_size)

    Y = function.affine(X_batch,W,B)
    E = function.error(Y,T_batch)
    E_save = np.append(E_save, E)

    Acc = function.accuracy(Y,T_batch)
    accuracy_save = np.append(accuracy_save, Acc)

    dW = function.delta_w(X_batch,T_batch,Y)
    dB = function.delta_b(T_batch,Y)

    W = function.update(W,learning_rate,dW)
    B = function.update(B,learning_rate,dB)

end_time = time.time()
total_time = end_time - start_time
print(total_time)

#show graph
function.plot_acc(accuracy_save)
function.plot_loss(E_save)