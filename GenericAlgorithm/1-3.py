from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import random
import math

population_number = 90
select_number = 10
# 첫 세대의 가중치를 랜덤으로 생성하고, 배열로 반환.
def random_weight_list():
    weight_list = list() # 리스트 선언
    for i in range(population_number):
        weight_list.append(random.uniform(-1,1)) # uniform 함수 이용해 랜덤으로 값 추가.
    return weight_list

# 첫 세대의 bias 를 랜덤으로 생성하고, 배열로 반환.
def random_bias_list():
    bias_list = list()
    for i in range(population_number):
        bias_list.append(random.uniform(-100,100)) # uniform 함수 이용해 랜덤으로 값 추가.
    return bias_list

# sigmoid 함수
def sigmoid(x):
    # math.exp(x) = e^x 와 같다.
    y = 1 / (1 + math.exp(-x))
    return y

# zeta 함수 값을 sigmoid 처리한 label 들을 반환하는 함수
def sigmoid_list(zeta_list):
    y = np.zeros((len(zeta_list), 1))  # 행렬 생성
    # zeta_list 의 개별 원소를 sigmoid 함수에 넣어서, y에 값 넣기
    for i in range(0, len(zeta_list)):
        y[i] = sigmoid(zeta_list[i])
        # y 반환
    return y

# zeta 함수
def zeta(w1, w2, w3, x1, x2, x3, b):
    z = np.zeros((100, 1))

    for i in range(100):
        z[i] = w1*x1[i] + w2*x2[i] + w3*x3[i] + b
    return z

# 유전자 교배
def crossover(w1_list, w2_list, w3_list, b_list):
    w1_next_generation = list()
    w2_next_generation = list()
    w3_next_generation = list()
    b_next_generation = list()

    w1_gene_list = list()
    w2_gene_list = list()
    w3_gene_list = list()
    b_gene_list = list()
    # 유전자 생성. 가중치 유전자는 각각 앞의 소수 5자리와 뒷소수 5자리로 나누어진다. bias 는 부모의 평균값을 가져옴.
    for i in range(select_number):
        w1_gene_list.append(weight_get_gene(w1_list[i]))
        w2_gene_list.append(weight_get_gene(w2_list[i]))
        w3_gene_list.append(weight_get_gene(w3_list[i]))
        b_gene_list.append(b_list[i])

    # 유전자 교배. 앞자리 + 뒷자리, b는 부모의 평균으로 구해온다.
    for i in range(select_number):
        for j in range(i+1, select_number):
            w1_next_generation.append(w1_gene_list[i][0] + w1_gene_list[j][1])
            w2_next_generation.append(w2_gene_list[i][0] + w2_gene_list[j][1])
            w3_next_generation.append(w3_gene_list[i][0] + w3_gene_list[j][1])
            b_next_generation.append((b_gene_list[i] + b_gene_list[j]) / 2)


    # 유전자 교배. 뒷자리 + 앞자리, b는 부모의 평균으로 구해온다.
    for i in range(select_number):
        for j in range(i+1, select_number):
            w1_next_generation.append(w1_gene_list[i][1] + w1_gene_list[j][0])
            w2_next_generation.append(w2_gene_list[i][1] + w2_gene_list[j][0])
            w3_next_generation.append(w3_gene_list[i][1] + w3_gene_list[j][0])
            b_next_generation.append((b_gene_list[i] + b_gene_list[j]) / 2)

    print("w1_next-generation :" , len(w1_next_generation))
    # 필요 이상의 유전자가 만들어졌으므로, 임의로 죽인다.
    for i in range(population_number - len(w1_next_generation)):
        kill_index = int(random.random() % 100)
        del w1_next_generation[kill_index]
        del w2_next_generation[kill_index]
        del w3_next_generation[kill_index]
        del b_next_generation[kill_index]

    # 튜플형으로 반환
    return [w1_next_generation, w2_next_generation, w3_next_generation, b_next_generation]

# 가중치 유전자를 tuple로 생성
def weight_get_gene(source):
    gene_a = ((source*100000) // 1) / 100000
    gene_b = source % 0.00001
    return gene_a, gene_b

# 출력해주는 함수
def visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1, w1, w2, w3, b, ax):
    ax.plot(x_0, y_0, z_0, linestyle="none", marker="o", mfc="none", markeredgecolor="r")  # 샘플 출력
    ax.plot(x_1, y_1, z_1, linestyle="none", marker="o", mfc="none", markeredgecolor="b")  # 샘플 출력

    X = np.arange(0, 2, 0.1) * 100
    Y = np.arange(0, 2, 0.1) * 100
    X, Y = np.meshgrid(X, Y)

    Z = (-float(w1) / w3 * X) + (-float(w2) / w3 * Y) - float(b) / w3  # 평면의 방정식

    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.4, cmap=cm.Blues)  # 평면 출력


data = pd.read_csv("p1_training_data.csv") # 데이터 읽기
np_data = np.array(data) # np Array 로 변경

fig = plt.figure() # 새로운 figure 생성
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')
ax3 = fig3.add_subplot(111, projection='3d')
ax4 = fig4.add_subplot(111, projection='3d')

## positive samples
x_1 = np_data[0:50,0]
y_1 = np_data[0:50,1]
z_1 = np_data[0:50,2]

## negative samples
x_0 = np_data[50:,0]
y_0 = np_data[50:,1]
z_0 = np_data[50:,2]

# 전체 data
x1 = np_data[0:,0]
x2 = np_data[0:,1]
x3 = np_data[0:,2]
label = np_data[0:,3]


# numpy 배열로 변경
w1_population = np.array(random_weight_list())
w2_population = np.array(random_weight_list())
w3_population = np.array(random_weight_list())
b_population = np.array(random_bias_list())

# index별로 fitness를 저장.
error_sum_list = list() # 오차 합을 (인덱스, 오차 합) 의 튜플 형식으로 저장한다.

# population 을 한 녀석씩 봄.
for learn_idx in range(population_number):
    # zeta 함수와 sigmoid 함수 이용해, 인공신경망이 산출해 낸 label 값 가져옴.
    zeta_list = zeta(w1_population[learn_idx], w2_population[learn_idx], w3_population[learn_idx], x1, x2, x3, b_population[learn_idx])
    learning_label_list = sigmoid_list(zeta_list)
    # fitness 검사과정
    error_sum = 0  # 임시로 오차 합을 저장할 변수

    # 오차합을 구하는 과정.
    for fit_idx in range(0, 100):
        error_sum += pow(learning_label_list[fit_idx] - label[fit_idx], 2)  # 오차합은 제곱을 해준다.

    error_sum_list.append((learn_idx, error_sum)) # 구한 오차합을, 현재 population 의 idx와 함께 저장해준다.


error_sum_list.sort(key=lambda x: x[1]) # 튜플을 두번째 원소를 이용해, 오름차순 정렬

# 부모 생성해 교배 시키기. 해당 리스트에는, 상위 14% 가 가지는 원소가 들어간다.
parent_w1 = list()
parent_w2 = list()
parent_w3 = list()
parent_b = list()

# 유전자 추가. 상위 14프로만 사용.
for i in range(select_number):
    parent_w1.append(w1_population[error_sum_list[i][0]])
    parent_w2.append(w2_population[error_sum_list[i][0]])
    parent_w3.append(w3_population[error_sum_list[i][0]])
    parent_b.append(b_population[error_sum_list[i][0]])

# 유전자 교배시키기.
next_population = crossover(parent_w1, parent_w2, parent_w3, parent_b)
# 새로운 세대
w1_population = next_population[0]
w2_population = next_population[1]
w3_population = next_population[2]
b_population = next_population[3]




# 그래프 각각의 fig에 추가함.
visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1, w1_population[error_sum_list[0][0]], w2_population[error_sum_list[0][0]], w3_population[error_sum_list[0][0]], b_population[error_sum_list[0][0]], ax)
visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1, w1_population[error_sum_list[1][0]], w2_population[error_sum_list[1][0]], w3_population[error_sum_list[1][0]], b_population[error_sum_list[1][0]], ax2)
visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1, w1_population[error_sum_list[2][0]], w2_population[error_sum_list[2][0]], w3_population[error_sum_list[2][0]], b_population[error_sum_list[2][0]], ax3)
visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1, w1_population[error_sum_list[3][0]], w2_population[error_sum_list[3][0]], w3_population[error_sum_list[3][0]], b_population[error_sum_list[3][0]], ax4)

plt.show()