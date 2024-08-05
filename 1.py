import numpy as np
import matplotlib.pylab as plt

d = 200
n = 200

p_max = 500

train = []
test = []
for p in range(1,p_max+1):

    x_list = []
    y_list = []
    W = np.random.rand(p,d)*2-1
    b = np.random.rand(p,1)*2-1
    # print(w,b)
    def f(x): #活性化関数
        return 1 / (1 + np.exp(-x))

        # return np.maximum(0, x)

        # exp_x = np.exp(x)
        # sum_exp_x = np.sum(exp_x)
        # y = exp_x / sum_exp_x
        # return y

    def model(x):
        e = np.random.normal(0,1,1)
        return b.T.dot(f(W.dot(x)))+e

    # x = np.random.rand(d,1)
    # print(model(x))

    for i in range(n):
        x = np.random.rand(d,1)*2-1
        y = model(x)
        x_list.append(x)
        y_list.append(y)

    X=x_list[0]
    Y=y_list[0]
    for i in range(1,n):
        X = np.append(X,x_list[i],axis=1)
        Y = np.append(Y,y_list[i],axis=0)

    # print(x_list)
    # print(y_list)
    # print(y_list[0][0][0])

    # print(f(x_list[0]))
    def beta():
        A = f(W.dot(X))
        A_plus = np.linalg.pinv(A)
        return Y.T.dot(A_plus).T

    # print(beta())

    def sigma(x):
        ret = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ii = x[i]-np.mean(x[i])
                jj = (x[j]-np.mean(x[j])).T
                ret[i][j] = ii.dot(jj)/x.shape[1]
        return ret

    # print(sigma(X))
    # def B():
    #     be = beta()
    #     return (be-b).T.dot(sigma(be-b))

    # def V():
    #     be = beta()
    #     c = (be-b).T.dot(sigma(be-b))
    #     return np.mean(sigma(c)) - B()
    # print(B(),V())
    # print(sigma(beta()-b))
    # print(b)

    def R():
        be = beta()
        c = (be-b).T.dot(sigma(be-b))
        return np.mean(sigma(c))
    
    def train_loss():
        y_pre = beta().T.dot(f(W.dot(X))).T
        return np.mean(np.linalg.norm(Y-y_pre))
    
    train.append(train_loss())

    x_test_list = []
    y_test_list = []
    for i in range(n):
        x = np.random.rand(d,1)*2-1
        y = model(x)
        x_test_list.append(x)
        y_test_list.append(y)

    X_test=x_test_list[0]
    Y_test=y_test_list[0]
    for i in range(1,n):
        X_test = np.append(X_test,x_test_list[i],axis=1)
        Y_test = np.append(Y_test,y_test_list[i],axis=0)

    def test_loss():
        y_pre = beta().T.dot(f(W.dot(X_test))).T
        return np.mean(np.linalg.norm(Y_test-y_pre))
    
    test.append(test_loss())

plt.plot(range(1,p_max+1), train, label="train", color="blue")
plt.plot(range(1,p_max+1), test, label="test", color="red")
plt.xlim(1, p_max)
# plt.ylim(0, 10.0)
plt.xlabel('Parameter')
plt.ylabel('Loss')
plt.legend()
plt.title("d="+str(d)+",n="+str(n))
plt.savefig("./images/1_"+str(d)+"_"+str(n)+".png")
plt.show()
plt.clf()