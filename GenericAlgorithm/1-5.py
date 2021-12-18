from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import random
import math

max_generation_number = 40 # 최대로 학습할 세대
population_number = 90 # 세대 당 개체 수
select_number = 10 # 선택할 개체의 갯수
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
    # 다음 세대로 전달될 w1, w2, w3, b
    w1_next_generation = list()
    w2_next_generation = list()
    w3_next_generation = list()
    b_next_generation = list()

    # 유전자를 임시로 저장할 리스트
    w1_gene_list = list()
    w2_gene_list = list()
    w3_gene_list = list()
    b_gene_list = list()
    # 유전자 생성. 가중치 유전자는 각각 앞의 소수 5자리와 뒷소수 5자리로 나누어진다. bias 는 부모의 평균값을 가져옴.
    # 자세한 사항은 레포트에 기재
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


    # 유전자 교배. 부모의 앞자리를 더한 평균 값. b도 마찬가지로 부모의 평균으로 구해온다.
    for i in range(select_number):
        for j in range(i+1, select_number):
            w1_next_generation.append((w1_gene_list[i][0] + w1_gene_list[j][0])/2)
            w2_next_generation.append((w2_gene_list[i][0] + w2_gene_list[j][0])/2)
            w3_next_generation.append((w3_gene_list[i][0] + w3_gene_list[j][0])/2)
            b_next_generation.append((b_gene_list[i] + b_gene_list[j]) / 2)

    # 혹시나 세대수를 변경할 시, 필요 이상의 유전자가 만들어지면, 임의로 제거한다.
    for i in range(population_number - len(w1_next_generation)):
        kill_index = int(random.random() % 100)
        del w1_next_generation[kill_index]
        del w2_next_generation[kill_index]
        del w3_next_generation[kill_index]
        del b_next_generation[kill_index]

    # 리스트형으로 반환
    return [w1_next_generation, w2_next_generation, w3_next_generation, b_next_generation]

# 가중치 유전자를 tuple로 생성후 반환한다.
# 유전자 생성은 a 는 소수점 5자리까지, b 는 소수점 5자리 뒤부터 이다.
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

fig_list = list()
for i in range(1):
    fig_list.append(plt.figure())

ax_list = list()
for i in range(1):
    ax_list.append(fig_list[i].add_subplot(111, projection="3d"))

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


# 세대 반복
for generation in range(max_generation_number):
    # population 을 한 녀석씩 봄.
    # index별로 fitness를 저장.
    error_sum_list = list()  # 오차 합을 (인덱스, 오차 합) 의 튜플 형식으로 저장한다.
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

    # 유전자 추가. 상위 14프로만 사용. 14프로만 사용하는 이유는 세대수 90과 교배규칙으로 나온 형질의 수를 맞추기 위해서
    for i in range(select_number):
        parent_w1.append(w1_population[error_sum_list[i][0]])
        parent_w2.append(w2_population[error_sum_list[i][0]])
        parent_w3.append(w3_population[error_sum_list[i][0]])
        parent_b.append(b_population[error_sum_list[i][0]])

    # 유전자 교배시키기.
    next_population = crossover(parent_w1, parent_w2, parent_w3, parent_b)
    # 새로운 세대교체
    w1_population = next_population[0]
    w2_population = next_population[1]
    w3_population = next_population[2]
    b_population = next_population[3]

    # 모든 개체는 1퍼센트의 확률로 변이할 수도 있다.
    for i in range(population_number):
        randomidx = random.randrange(0, 100)
        if randomidx==1:
            print("mutation occur") # mutation이 발생할 경우 알려주는 문구.
            # 임의의 변이 규칙에 의해 개체에 변이가 일어난다.
            w1_population[i] = w1_population[i] + 0.2
            w2_population[i] = w2_population[i] - 0.2
            w3_population[i] = w3_population[i] / 2
            b_population[i] = b_population[i] - 2
    print("--- %d 세대 의 fitness score ---" %(generation+1))
    print("1st fitness score =", error_sum_list[0][1])
    print("2nd fitness score =", error_sum_list[1][1])
    print("3rd fitness score =", error_sum_list[2][1])
    print("4th fitness score =", error_sum_list[3][1])

    # 각 세대에서 선택될 개체들 중에서, 가장 fitness가 낮은 10번째 개체의 fitness의 값이 0.0001보다 작으면, 정확하게 분류했다 판단하고 종료합니다.
    if error_sum_list[9][1] < 0.0001:
        print("이번 세대의 10번째 개체의 fitness :", error_sum_list[9][1])
        break



# best fitness 를 가지는 것 시각화.
visualize_grid(x_0, y_0, z_0, x_1, y_1, z_1, w1_population[error_sum_list[0][0]],
                       w2_population[error_sum_list[0][0]], w3_population[error_sum_list[0][0]],
                       b_population[error_sum_list[0][0]], ax_list[0])

plt.show()