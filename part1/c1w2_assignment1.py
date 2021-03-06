import numpy as np
import time
def L1(yhat,y):
    print(y.shape)
    print(yhat.shape)
    loss = np.sum(np.abs(y-yhat),axis=0)
    print(loss)
    return loss

def L2(yhat,y):
    loss = np.dot(y-yhat,y-yhat).T
    print(loss)
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L2(yhat,y)))


# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
#
# ### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
# tic = time.process_time()
# dot = 0
# for i in range(len(x1)):
#     dot+= x1[i]*x2[i]
# toc = time.process_time()
# print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#
# ### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
# tic = time.process_time()
# outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
# for i in range(len(x1)):
#     for j in range(len(x2)):
#         outer[i,j] = x1[i]*x2[j]
# toc = time.process_time()
# print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#
# ## CLASSIC ELEMENTWISE IMPLEMENTATION ###
# tic = time.process_time()
# mul = np.zeros(len(x1))
# for i in range(len(x1)):
#     mul[i] = x1[i]*x2[i]
# toc = time.process_time()
# print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#
# ### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
# W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
# tic = time.process_time()
# gdot = np.zeros(W.shape[0])
# for i in range(W.shape[0]):
#     for j in range(len(x1)):
#         gdot[i] += W[i,j]*x1[j]
# toc = time.process_time()
# print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#
# x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
# x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
#
# ### VECTORIZED DOT PRODUCT OF VECTORS ###
# tic = time.process_time()
# dot = np.dot(x1,x2)
# toc = time.process_time()
# print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#
# ### VECTORIZED OUTER PRODUCT ###
# tic = time.process_time()
# outer = np.outer(x1,x2)
# toc = time.process_time()
# print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

# ### VECTORIZED ELEMENTWISE MULTIPLICATION ###
# tic = time.process_time()
# mul = np.multiply(x1,x2)
# toc = time.process_time()
# print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
#
# ### VECTORIZED GENERAL DOT PRODUCT ###
# tic = time.process_time()
# dot = np.dot(W,x1)
# toc = time.process_time()
# print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")










# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def sigmoid_derivarite(x):
#     s = sigmoid(x)
#     ds = s*(1-s)
#     return ds
# def image2vector(images):
#     print(images.reshape(images.shape[0]*images.shape[1]*images.shape[2],1))
#
#
# def normalizeRows(x):
#     x_norm = np.linalg.norm(x, axis=1,keepdims=True)
#     print(x_norm)
#     x = x / x_norm
#     return x
# def softmax(x):
#     x = np.exp(x)
#     sumx = np.sum(x,axis=1,keepdims=True)
#     return x / sumx
#
#
# if __name__ == '__main__':
#     x = np.array([
#         [9, 2, 5, 0, 0],
#         [7, 5, 0, 0, 0]])
#     print("softmax(x) = " + str(softmax(x)))
    # x = np.array([
    #     [0, 3, 4],
    #     [1, 6, 4]])
    # print("normalizeRows(x) = " + str(normalizeRows(x)))
   # x = np.array([1,2,3])
   # print(sigmoid(x))
   # print(sigmoid_derivarite(x))

   # images = np.array([[[0.67826139, 0.29380381],
   #                    [0.90714982, 0.52835647],
   #                    [0.4215251, 0.45017551]],
   #
   #                   [[0.92814219, 0.96677647],
   #                    [0.85304703, 0.52351845],
   #                    [0.19981397, 0.27417313]],
   #
   #                   [[0.60659855, 0.00533165],
   #                    [0.10820313, 0.49978937],
   #                    [0.34144279, 0.94630077]]])
   #
   # print("image2vector(images) = " + str(image2vector(images)))
