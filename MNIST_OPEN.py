import pickle

save_file = '/Users/FUTOSHI/Desktop/MNIST_test/mnist.pkl'

with open(save_file, 'rb') as f:
    dataset = pickle.load(f)

train_img,train_label,test_img,test_label = dataset

#show image
import matplotlib.pyplot as plt

n = 1000
img = train_img[n].reshape((28, 28))

plt.imshow(img)
plt.show()

print(train_label[n])