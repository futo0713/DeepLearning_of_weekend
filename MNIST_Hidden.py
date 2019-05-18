import function1
import numpy as np
import matplotlib.pyplot as plt
import time

#minist_load
save_file = '/Users/FUTOSHI/Desktop/MNIST_test/mnist.pkl'
X_train,T_train,X_test,T_test = function1.mnist(save_file)

# function_ver1.show(X_train,T_train,0)

#initial setting
num_of_hidden=100
Wh = np.random.randn(784, num_of_hidden)
Bh = np.random.randn(1,num_of_hidden)
Wo = np.random.randn(num_of_hidden, 10)
Bo = np.random.randn(1,10)
learning_rate = 0.001

E_save = []
accuracy_save = []
start_time = time.time()

batch_size = 100

#iteration
num_of_itr=3000
for i in range(num_of_itr):

    X_batch,T_batch = function1.batch(X_train,T_train,batch_size)

    H = function1.affine(X_batch,Wh,Bh,'sigmoid')
    Y = function1.affine(H,Wo,Bo,'softmax')

    E = function1.error(Y,T_batch)
    E_save = np.append(E_save, E)

    Acc = function1.accuracy(Y,T_batch)
    accuracy_save = np.append(accuracy_save, Acc)

    #=======================
    dWo = np.dot(H.T,(Y-T_batch))
    dBo = np.reshape(np.sum(Y-T_batch, axis=0),(1,10))

    dWh = np.dot(X_batch.T,H*(1-H)*np.dot(Y-T_batch,Wo.T))
    dBh = np.sum(H*(1-H)*np.dot(Y-T_batch,Wo.T), axis=0, keepdims=True)
    #=======================
    
    Wh = function1.update(Wh,learning_rate,dWh)
    Bh = function1.update(Bh,learning_rate,dBh)
    Wo = function1.update(Wo,learning_rate,dWo)
    Bo = function1.update(Bo,learning_rate,dBo)


end_time = time.time()
total_time = end_time - start_time
print(total_time)

#show graph
function1.plot_acc(accuracy_save)
function1.plot_loss(E_save)