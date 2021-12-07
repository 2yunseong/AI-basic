from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sym
import numpy as np

f = open("Linear_classification.txt", 'r')  # 파일 오픈
lines = f.readlines()   # 각 데이터서 개행 단위로 요소를 만들어서 list로 바꿈
data_list = []         # data들을 저장할 리스트

# linear classification 에서 읽은 파일을 data_list에 옮기는 과정
for line in lines:
    line = line.strip()  # 개행 문자 제거
    temp_list = line.split(' ')  # 공백글자에 따라 데이터를 나눔
    data_list.append(temp_list)  # data_list에 추가

# 사용할 Symbol 정의
a = sym.Symbol('a')
b = sym.Symbol('b')
c = sym.Symbol('c')
z = sym.Symbol('z')
x = sym.Symbol('x')
y = sym.Symbol('y')

# 최소 자승법으로 해 구하기


def least_square_method(datalist):
    # 오차의 합을 저장할 변수 선언
    error_summation_a = 0*a
    error_summation_b = 0*b
    error_summation_c = 0*c

    # 오차 누적합
    for element in data_list:
        # 오차 구하기
        element_error = int(element[1])*a + \
            int(element[2])*b + c - int(element[3])
        element_error = element_error*element_error  # 오차 제곱
        # Ra 미분, z가 0 이므로 미분한 값만 더해주면 됨.
        de_element_error_a = sym.diff(element_error, a)
        # 누적 합 변수에 더해준다.
        error_summation_a = error_summation_a + de_element_error_a

        # b 와 c에도 위 과정 반복
        de_element_error_b = sym.diff(element_error, b)
        error_summation_b = error_summation_b + de_element_error_b

        de_element_error_c = sym.diff(element_error, c)
        error_summation_c = error_summation_c + de_element_error_c

    # 구한 R 을 행렬계산을 위해 계수를 추출하는 과정.
    # 다항식으로 바꿔준 후,
    coeffi_Ra = sym.Poly(error_summation_a, a, b, c)
    coeffi_Rb = sym.Poly(error_summation_b, a, b, c)
    coeffi_Rc = sym.Poly(error_summation_c, a, b, c)

    # 각각 리스트를 선언하고 계수를 저장해준다.
    coeffi_Ra_list = coeffi_Ra.coeffs()
    coeffi_Rb_list = coeffi_Rb.coeffs()
    coeffi_Rc_list = coeffi_Rc.coeffs()

    # 행렬 계산을 하기위해 각각의 리스트를 만들어 줌
    matrix_x = [float(coeffi_Ra_list[0]), float(
        coeffi_Rb_list[0]), float(coeffi_Rc_list[0])]
    matrix_y = [float(coeffi_Ra_list[1]), float(
        coeffi_Rb_list[1]), float(coeffi_Rc_list[1])]
    matrix_z = [float(coeffi_Ra_list[2]), float(
        coeffi_Rb_list[2]), float(coeffi_Rc_list[2])]
    matrix_b = [(-1)*float(coeffi_Ra_list[3]), (-1) *
                float(coeffi_Rb_list[3]), (-1)*float(coeffi_Rc_list[3])]

    # 3x3 행렬
    matrix_A = [matrix_x, matrix_y, matrix_z]

    inv_A = np.linalg.inv(matrix_A)  # 역행렬 구하기
    result = np.dot(inv_A, matrix_b)  # 구하려는 해

    # 결과 리턴
    return result


# 위에서 정의한 최소자승법으로 구한 모델의 계수를 리스트 형식으로 저장함.
linear_classification_coeffs = least_square_method(data_list)

# 해당의 해
linear_equation_answer = linear_classification_coeffs[0] * \
    x + linear_classification_coeffs[1]*y + linear_classification_coeffs[2] - z


sample_x = []   # 다리길이
sample_y = []   # 팔길이
sample_z = []   # 성별 (수컷=1 암컷=-1)

# sample data 추가
for ele in data_list:
    sample_x.append(int(ele[1]))
    sample_y.append(int(ele[2]))
    sample_z.append(int(ele[3]))

# 자료실 코드 참조
X = np.arange(-0.5, 1.5, 0.1)*100
Y = np.arange(-0.5, 1.5, 0.1)*100
X, Y = np.meshgrid(X, Y)

# X, Y 바탕으로 평면의 방정식 구현
Z = (float(linear_classification_coeffs[1]) * Y) + (float(
    linear_classification_coeffs[0]) * X) + float(linear_classification_coeffs[2])

# 자료실 코드 참조
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# x 값은 -50~ 150까지 임의로 주고, 평면의 방정식에서 z는 0 이므로 ax + by + c = 0 으로 나타낼 수 있다.
# 방정식을 y에 관해 정리하면 다음과 같은 식을 얻을 수 있다. ( y = -(a/b)x - (c/b) )
equation_x = np.array(range(-50, 150))
equation_y = (-1)*(linear_classification_coeffs[0]/linear_classification_coeffs[1])*equation_x - (
    linear_classification_coeffs[2]/linear_classification_coeffs[1])

ax.plot(sample_x, sample_y, sample_z, linestyle="none", marker="o",
        mfc="none", markeredgecolor="k")  # 샘플 출력
ax.plot_surface(X, Y, Z, rstride=4, cstride=4,
                alpha=0.4, cmap=cm.Blues)  # 평면 출력
ax.plot(equation_x, equation_y, 'r-')  # 선 출력
plt.show()
f.close()  # 파일 닫기
