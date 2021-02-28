import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Tiền xử lý 
#chuyển ma trận 209x64x64x3 thành ma trận 209x12287x1

train_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_y = train_set_y.reshape(1, train_set_y.shape[1])
test_y = test_set_y.reshape(1, test_set_y.shape[1])
#chuẩn hóa dữ liệu bằng cách chia tất cả dữ liệu cho 255
print(train_x)
train_x = train_x/255.0
test_x = test_x/255.0

w = np.zeros(shape=(train_x.shape[0],1))
b = 0
print("len w: " + str(w.shape))

alpha = 1
train_time = 10000
m = train_x.shape[0]

def sigmoid(z):
    s =  1/(1+np.exp(-z))
    return s

def propagate(w, b, X, Y):

    z = np.dot(w.T, train_x) + b
    A = sigmoid(z)
    loss = (-1/m)*np.sum(train_y*np.log(A)+(1-train_y)*np.log(1-A))

    dz = A - train_y
    dw = (1/m)*np.dot(train_x,dz.T)
    db = (1/m)*np.sum(dz)
    
    return dw, db, loss
loss = 0.
for i in range(train_time):
    dw, db , loss = propagate(w, b, train_x, train_y)
    w = w - alpha*dw
    b = b - alpha*db
    
print("b: "+ str(b))
print("w: " + str(w))
print("w: " + str(w.shape))
loss *=100;
print("loss: " + str(loss) + "%")

from numpy import save


save('w.npy', w)
b.tofile('b.bin')

from numpy import load

w_store = np.load("w.npy", allow_pickle=True,encoding='ASCII')
b_store = np.fromfile('b.bin')[0]
#w_store = w_store.reshape(12288, 209)
print("b_store: " + str(b_store))
print("w_store: " + str(w_store))
print("w_len: " + str(w_store.shape))
index = 1
for i in range(50):
    index = i
    X = test_x.T[index]
    plt.imshow(test_set_x_orig[index])
    print(X.shape)

    predict = sigmoid(np.dot(w_store.T, X) + b)
    print(predict)
    if(predict > 0.5):
        plt.title("it's a cat"+str(predict))
        print(test_y[0][index])
    else:
        plt.title("It's not a cat"+str(predict))
        print(test_y[0][index])
    plt.show()