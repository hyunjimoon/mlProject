# Day_03_04_cost_gradient.py

def cost(x, y, W):
    loss = 0
    for i in range(len(x)):
        hx = W * x[i]
        loss += (hx - y[i]) ** 2
    return loss / len(x)

def gradient_descent(x, y, W):
    grad = 0
    for i in range(len(x)):
        grad += (W*x[i] - y[i]) * x[i]
    return grad / len(x)

x = [1, 2, 3]
y = [1, 2, 3]

W = 100
for i in range(100):
    loss = cost(x, y, W)
    grad = gradient_descent(x, y, W)
    W -= grad*0.1

    # early stop
    if loss < 1.0e-15:
        break

    print('{:2} : {:.6f}  {:.15f}'.format(i, W, loss))











