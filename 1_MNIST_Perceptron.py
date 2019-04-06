import numpy as np
import matplotlib.pyplot as plt
import time

#load dataset
import pickle

save_file = '/Users/Futoshi/Desktop/MNIST_test/mnist.pkl'

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

train_img,train_label,test_img,test_label = dataset

#pixel normalization
X_train, X_test = train_img/255, test_img/255

T_train = np.eye(10)[list(map(int,train_label))] 
T_test = np.eye(10)[list(map(int,test_label))] 

#function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def loss(y, t):
    delta = 1e-7
    return -np.sum(np.multiply(t, np.log(y+delta)) + np.multiply((1 - t), np.log(1 - y+delta)))


W = np.random.randn(784, 10)
B = np.random.randn(1,10)
learning_rate = 0.001

E_save = []
accuracy_save = []

#iteration
start_time = time.time()

num_of_itr=3000
for i in range(num_of_itr):
    #set batch
    train_size = X_train.shape[0] #60000
    batch_size = 100
    batch_mask = np.random.choice(train_size,batch_size)

    X_batch = X_train[batch_mask]
    T_batch = T_train[batch_mask]

    #forward prop
    Y = softmax(np.dot(X_batch, W)+B)
    E = loss(Y,T_batch)/len(Y)
    E_save = np.append(E_save, E)

    #set accuracy
    Y_accuracy = np.argmax(Y, axis=1)
    T_accuracy = np.argmax(T_batch, axis=1)
    accuracy = 100*np.sum(Y_accuracy == T_accuracy)/batch_size

    accuracy_save = np.append(accuracy_save, accuracy)

    #back prop
    dW = np.dot(X_batch.T,(Y-T_batch))
    dB = np.reshape(np.sum(Y-T_batch, axis=0),(1,10))

    W = W - learning_rate*dW
    B = B - learning_rate*dB

end_time = time.time()
time = end_time - start_time
print(time)

#plot
plt.figure()
plt.title('ACCURACY')
plt.xlabel("LEARNING NUMBER(EPOCH)")
plt.ylabel("ACCURACY (%)")
plt.xlim(0, 3000)
plt.ylim(0, 100)
plt.grid(True)
plt.plot(accuracy_save, color='blue')
plt.show()

plt.figure()
plt.title('LOSS FUNCTION')
plt.xlabel("LEARNING NUMBER(EPOCH)")
plt.ylabel("LOSS VALUE")
plt.xlim(0, 3000)
# plt.ylim(0, 50)
plt.grid(True)
plt.plot(E_save, color='blue')
plt.show()