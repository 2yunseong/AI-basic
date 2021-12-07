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

    # 오차의 누적 합
    for element in data_list:
        # 오차 구하기
        element_error = int(element[1])*a + \
            int(element[2])*b + c - int(element[3])
        element_error = element_error*element_error  # 오차 제곱
        # Ra 미분, z가 0이므로 미분한 값만 더해주면 됨
        de_element_error_a = sym.diff(element_error, a)
        # 누적합 변수에 더해준다.
        error_summation_a = error_summation_a + de_element_error_a

        # b와 c에도 위 과정 반복
        de_element_error_b = sym.diff(element_error, b)
        error_summation_b = error_summation_b + de_element_error_b

        de_element_error_c = sym.diff(element_error, c)
        error_summation_c = error_summation_c + de_element_error_c

    # 구한 R 을 행렬계산을 위해 계수를 추출하는 과정.
    # 다항식으로 바꿔준 후,
    coeffi_Ra = sym.Poly(error_summation_a, a, b, c)
    coeffi_Rb = sym.Poly(error_summation_b, a, b, c)
    coeffi_Rc = sym.Poly(error_summation_c, a, b, c)

    # 각각 리스트를 선언하고 저장해준다.
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

    # 리턴
    return result


# 위에서 정의한 최소자승법으로 구한 모델의 계수를 리스트 형식으로 저장함.
linear_classification_coeffs = least_square_method(data_list)

# sympy 를 이용해서, 방정식 꼴로 바꿔줌. ax+by+c=z 에서,z를 이항해주고, ax+by+c-z = 0 꼴로 나타내, 좌변만 출력함.
# linear_equation_answer = linear_classification_coeffs[0]*x + linear_classification_coeffs[1]*y + linear_classification_coeffs[2] - z

# 팔, 다리 길이 정의
arm_length = 50
leg_length = 30

# 각 x, y 에 다리, 팔 값을 넣어 줌. 여기서 나오는 z값으로 암컷인지 수컷인지 분류 가능.
z_value = float(linear_classification_coeffs[0])*leg_length + \
    float(linear_classification_coeffs[1])*arm_length + \
    float(linear_classification_coeffs[2])

sex = ""    # 성별 변수

# 성별 정하기
if z_value > 0:
    sex = "수컷"
elif z_value < 0:
    sex = "암컷"

# 답안 출력
print("입력한 동물은 학습 모델에 의해 " + sex + "으로 판단 됩니다.")


f.close()  # 파일 닫기
