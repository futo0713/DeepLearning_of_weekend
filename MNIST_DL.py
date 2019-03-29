import urllib.request

url_base = 'http://yann.lecun.com/exdb/mnist/'

dl_list = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

dataset_dir = '/Users/FUTOSHI/Desktop/MNIST_test'

for i in dl_list:
    file_path = dataset_dir + '/' + i 
    urllib.request.urlretrieve(url_base + i, file_path)