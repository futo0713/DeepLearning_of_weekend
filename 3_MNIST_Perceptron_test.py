#-----------------------------------------------------
#load dataset
import pickle

save_file = '/Users/tsutsumifutoshishi/Desktop/MNIST_test/mnist.pkl'

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

train_img,train_label,test_img,test_label = dataset


save_file = '/Users/tsutsumifutoshishi/Desktop/MNIST_test/Perceptron_300000.pkl'

with open(save_file, 'rb') as f:
    parameters = pickle.load(f)

W, B = parameters

#-----------------------------------------------------
#library import
import numpy as np
import matplotlib.pyplot as plt
import time

#-----------------------------------------------------
#pixel normalization
X_train, X_test = train_img/255, test_img/255

#transform_OneHot
T_train = np.eye(10)[list(map(int,train_label))] 
T_test = np.eye(10)[list(map(int,test_label))]

#-----------------------------------------------------
#function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def loss(y, t):
    delta = 1e-7
    return -np.sum(np.multiply(t, np.log(y+delta)) + np.multiply((1 - t), np.log(1 - y+delta)))

#-----------------------------------------------------
# for i in range(X_test[0]):

correct_img = np.empty((0,784))
error_img = np.empty((0,784))


start_time = time.time()
num_of_test = X_test.shape[0]
for i in range(num_of_test):
    #-----------------------------------------------------
    #forward prop
    Y = softmax(np.dot(X_test[i], W)+B)

    #set accuracy
    Y_accuracy = np.argmax(Y)
    T_accuracy = np.argmax(T_test[i])

    if Y_accuracy == T_accuracy:
        correct_img = np.vstack((correct_img,X_test[i]))

    else:
        error_img = np.vstack((error_img,X_test[i]))

end_time = time.time()

print(correct_img.shape)
print(error_img.shape)
print(100*correct_img.shape[0]/num_of_test)
print(end_time - start_time)