#-----------------------------------------------------
#load dataset
import pickle

save_file = '/Users/FUTOSHI/Desktop/MNIST_test/mnist.pkl'

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

train_img,train_label,test_img,test_label = dataset

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
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def loss(y, t):
    delta = 1e-7
    return -np.sum(np.multiply(t, np.log(y+delta)) + np.multiply((1 - t), np.log(1 - y+delta)))

#-----------------------------------------------------
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

#-----------------------------------------------------
#iteration
num_of_itr=3000
for i in range(num_of_itr):

    #-----------------------------------------------------
    #set batch
    train_size = X_train.shape[0]
    batch_size = 100
    batch_mask = np.random.choice(train_size,batch_size)

    X_batch = X_train[batch_mask]
    T_batch = T_train[batch_mask]

    #-----------------------------------------------------
    #forward prop
    H = sigmoid(np.dot(X_batch,Wh)+Bh)
    Y = softmax(np.dot(H, Wo)+Bo)
    E = loss(Y,T_batch)/len(Y)
    E_save = np.append(E_save, E)

    #-----------------------------------------------------
    #set accuracy
    Y_accuracy = np.argmax(Y, axis=1)
    T_accuracy = np.argmax(T_batch, axis=1)
    accuracy = 100*np.sum(Y_accuracy == T_accuracy)/batch_size

    accuracy_save = np.append(accuracy_save, accuracy)
 
    #-----------------------------------------------------
    #back prop
    dWo = np.dot(H.T,(Y-T_batch))
    dBo = np.reshape(np.sum(Y-T_batch, axis=0),(1,10))

    dWh = np.dot(X_batch.T,H*(1-H)*np.dot(Y-T_batch,Wo.T))
    dBh = np.sum(H*(1-H)*np.dot(Y-T_batch,Wo.T), axis=0, keepdims=True)


    Wo = Wo - learning_rate*dWo
    Bo = Bo - learning_rate*dBo
    Wh = Wh - learning_rate*dWh
    Bh = Bh - learning_rate*dBh

end_time = time.time()
time = end_time - start_time
print(time)

#-----------------------------------------------------
#plot
plt.figure()
plt.title('accuracy')
plt.xlabel("LEARNING NUMBER(EPOCH)")
plt.ylabel("ACCURACY (%)")
plt.xlim(0, 3000)
plt.ylim(0, 100)
plt.grid(True)
plt.plot(accuracy_save, color='blue')
plt.savefig('hidden_figure(accuracy)')
plt.show()

plt.figure()
plt.title('LOSS FUNCTION')
plt.xlabel("LEARNING NUMBER(EPOCH)")
plt.ylabel("LOSS VALUE")
# plt.xlim(0, 3000)
# plt.ylim(0, 100)
plt.grid(True)
plt.plot(E_save, color='blue')
plt.savefig('hidden_figure(loss)')
plt.show()

#-----------------------------------------------------
#Save parameters

parameters = [Wh,Bh,Wo,Bo]

dataset_dir = '/Users/FUTOSHI/Desktop/MNIST_test'

save_file = dataset_dir + '/hidden.pkl'

with open(save_file, 'wb') as f:
    pickle.dump(parameters, f) 