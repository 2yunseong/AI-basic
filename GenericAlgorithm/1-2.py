from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import random
import math

# 첫 세대의 가중치를 랜덤으로 생성하고, 배열로 반환.
def random_weight_list():
    weight_list = list() # 리스트 선언
    for i in range(70):
        weight_list.append(random.uniform(-1,1)) # uniform 함수 이용해 랜덤으로 값 추가.
    return weight_list

# 첫 세대의 bias 를 랜덤으로 생성하고, 배열로 반환.
def random_bias_list():
    bias_list = list()
    for i in range(70):
        bias_list.append(random.uniform(-100,100)) # uniform 함수 이용해 랜덤으로 값 추가.
    return bias_list

# sigmoid 함수
def sigmoid(x):
    # math.exp(x) = e^x 와 같다.
    y = 1 / (1 + math.exp(-x))
    return y

# 출력해주는 함수
def visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1):
    ax.plot(x_0, y_0, z_0, linestyle="none", marker="o", mfc="none", markeredgecolor="r")  # 샘플 출력
    ax.plot(x_1, y_1, z_1, linestyle="none", marker="o", mfc="none", markeredgecolor="b")  # 샘플 출력

    X = np.arange(0, 2, 0.1) * 100
    Y = np.arange(0, 2, 0.1) * 100
    X, Y = np.meshgrid(X, Y)

    Z = (-float(w1) / w3 * X) + (-float(w2) / w3 * Y) - float(b) / w3  # 평면의 방정식

    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.4, cmap=cm.Blues)  # 평면 출력
    plt.show()

data = pd.read_csv("p1_training_data.csv") # 데이터 읽기
np_data = np.array(data) # np Array 로 변경

fig = plt.figure() # 새로운 figure 생성
ax = fig.add_subplot(111, projection='3d')

## positive samples
x_1 = np_data[0:50,0]
y_1 = np_data[0:50,1]
z_1 = np_data[0:50,2]
postive_label_sample = np_data[0:50,3]

## negative samples
x_0 = np_data[50:,0]
y_0 = np_data[50:,1]
z_0 = np_data[50:,2]

## Generation 1의 fittest gene
w1 = 0.28645574
w2 = -0.43628723
w3 = 0.30481866
b = -14.39337271

# numpy 배열로 변경
w1_population = np.array(random_weight_list())
w2_population = np.array(random_weight_list())
w3_population = np.array(random_weight_list())
b_population = np.array(random_bias_list())



visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1)

plt.show()