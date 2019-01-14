import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # numpy의 브로드캐스트

def identity_function(x):
    # 출력층의 활성화 함수는 풀고하 하는 문제의 성질에 맞게 정의
    # 예를 들어 회귀에는 항등 함수, 2클래스 분류에는 시그모이드 함수, 다중 클래스 분류에는 소프트맥수 함수 등
    return x

def init_network():
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    network = [ [W1,B1],[W2,B2],[W3,B3] ]
    return network

def forward(network, x, out):
    for idx in range(len(network)):
        if idx==0:
            a = np.dot(x, network[idx][0]) + network[idx][1] # x로 시작
            out.append(sigmoid(a))
            continue
        if idx==len(network)-1:
            a = np.dot(out[-1], network[idx][0]) + network[idx][1]
            out.append(identity_function(a)) # identity function으로 끝
            return out[-1] # list를 통째로 내보내는게 더 유연할 듯
        a = np.dot(out[-1], network[1][0]) + network[1][1]
        out.append(sigmoid(a))

network = init_network()
out = []

x = np.array([1.0, 0.5])
y = forward(network,x,out)
print(y)
