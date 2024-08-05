import numpy as np
import matplotlib.pylab as plt

n = 60
p = 2*4*4*4*4*4 #p>n

x_list = []
y_list = []

def model(x_data):
    e = np.random.default_rng().normal(0,0.4)
    return np.cos(3*x_data)+e

for i in range(n):
    x_data = np.random.rand(1,1)*np.pi
    y_data = model(x_data)
    x_list.append(x_data)
    y_list.append(y_data)

X=x_list[0]
Y=y_list[0]
for i in range(1,n):
    X = np.append(X,x_list[i],axis=1)
    Y = np.append(Y,y_list[i],axis=0)

# def phi(x):
#     ret = np.cos(x)
#     for i in range(2,p+1):
#         ret = np.append(ret, np.cos((i+1)*x), axis=0)
#     return ret

def phi(x):
    ret = np.cos(x)
    for i in range(2,p+1):
        ret = np.append(ret, (1/i)*np.cos(i*x), axis=0)
    return ret
    
def beta():
    A = phi(X)
    A_plus = np.linalg.pinv(A)
    return Y.T.dot(A_plus).T
    
b = beta()

def predict(x):
    return b.T.dot(phi(x))

xx=np.zeros(1000).reshape(1,1000)
for i in range(1000):
    xx[0][i] = i*0.001*np.pi

cos = [[]]
result = []
cos = np.cos(3*xx)
result = predict(xx)

np_x = np.array(x_list).T[0][0]
np_y = np.array(y_list).T[0][0]
sorted_index = np.argsort(np_x)
sorted_x = np_x[sorted_index]
sorted_y = np_y[sorted_index]

plt.plot(xx[0], cos[0], label="y=cos(3x)", color="green")
plt.plot(xx[0], result[0], label="predict", color="red")
for i in range(n):
    plt.plot(sorted_x[i], sorted_y[i], marker='.', markersize=5, color="blue")
plt.xlim(0,3.14)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("n="+str(n)+",p="+str(p))
plt.savefig("./images/2_second_"+str(n)+"_"+str(p)+".png")
plt.show()
plt.clf()