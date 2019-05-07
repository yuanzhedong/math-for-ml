import math
import numpy as np

def mean(x):
    sum = 0
    for i in x:
        sum += i
    return sum * 1.0 / len(x)

def var(x):
    sum = 0
    m = mean(x)
    for i in x:
        sum += (i - m) ** 2
    return sum * 1.0 / len(x)


x = [1, 2, 3, 2]
print(var(x))
print(math.sqrt(var(x)))

print(math.acos(1.0 * -7 / 5 / math.sqrt(2)))
print(math.sqrt(13))
print(math.acos(9 / math.sqrt(14) / math.sqrt(33)))

a = np.array([1, -1, 3])

b = np.array([[2,1,0],[1,2,-1],[0,-1,2]])

x = a @ b
print(x.shape)
print(a.T @ b @ a)

x = np.array([0.5, -1, -0.5])
y = np.array([0, 1, 0])

b = np.array([[2, 1, 0], [1, 2, -1], [0, -1, 2]])

print((x-y)@ b @ (x-y))

x = np.array([4, 2, 1])
y = np.array([0, 1, 0])

print(math.sqrt((x - y) @ b @ (x - y)))

print(math.acos(-1.0/ 3))

x = np.array([0, -1])
y = np.array([1, 1])

b = np.array([[1, -1.0/2], [-1.0/2, 5]])

print(x @ b @ y)

print(x @ b @ x)
print(y @ b @ y)
print(math.acos(-0.9))


x = np.array([1, 1])
y = np.array([1, -1])

b = np.array([[1, 0], [0, 5]])

print(math.acos((x @ b @ y) / math.sqrt(x @ b @ x) / math.sqrt(y @ b @ y))) 


print(math.acos(1.0))

dist = [2, 3, 4, 1]
y = np.array([1, 2, 3, 4])

print(y)
print(np.argsort(dist))
print(y[np.argsort(dist)])

x = [[True, False, True, True, False, True]]

print(np.count_nonzero(x))


x = np.array([[0,1,1,1,1,2,2,1,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,2]])
print(np.count_nonzero(x[0] == 1))

print(x[0] == 1)


x = np.array([[2, 3, 4], [7, 8, 8]])
print(np.argsort(x))

dist = [[3.08058436],[2.56124969]]
print(np.argsort(dist))

x = np.zeros((5, 1))
print(np.linalg.norm(np.array([2, 3]) - np.array([1,2])))
x[0] = np.linalg.norm(np.array([2, 3]) - np.array([1,2]))
print(x)
print(np.argsort(x.flatten()))

print(math.sqrt(18)/9)

from numpy.linalg import inv
B = np.array([[1,0], [1, 1], [1, 2]])

x = np.array([12, 0, 0]).reshape((3, 1))
#print(x.shape, B.shape)
pm = B @ inv(B.T @ B) @ B.T
print(pm.shape)
print(pm @ x)

# lda = inv(B.T @ B) @ B.T @ x
# print(lda)


x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
f = np.ravel(x)
print(f)
print(f.reshape((4, 2), order='F'))


import numpy as np
import numpy.linalg as linalg

A = np.random.random((3,3))
eigenValues, eigenVectors = linalg.eig(A)

print(eigenValues)
print(eigenValues.argsort())
idx = eigenValues.argsort()[::-1]

print(eigenVectors)
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print(eigenVectors)
