------

------

## Generic Algorithm

------

### 개요

------

설계하려는 Generic Algorithm의 목표는 3차원 상의 점들을 linear classification하는 평면을 찾는 Generic Algorithm을 설계하는 것입니다. dataset은 각각 x1, x2, x3의 input data와 label값인 y 값을 가집니다. y는 0또는 1의 값을 가집니다. **single-layer neural network** 을 사용하고, activation function 은 sigmoid function을 사용합니다.

population수는 모든 세대가 같게 합니다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/49d8e121-b2e8-4f9e-897a-daf759f55dad/Untitled.png)

### 초기 세대 설정

------

초기 세대는 임의로 지정됩니다. Python의 random 모듈을 사용해 초기화 하였습니다.  단, weight 값은 -1 ~ 1 사이의 값을 가지고, bias는 -100 과 100 사이의 값을 가집니다. 이러한 개체의 수는 90마리로 설정합니다.

### Selection 규칙

------

selection은 fitness를 통해 판별합니다. 다음 세대로 이어나갈 10마리의 개체를 선정합니다.

fitness는 다음과 같이 구합니다.

```
모든 개체의 (실제 label(y값) - 모델이 추측한 label)^2 의 합
```

### CrossOver 규칙

------

Crossover는 다음과 같은 과정을 통해 수행됩니다.

**1) weight 유전자**

소수부 앞자리와 뒷자리로 나누는데, 앞자리는 무조건 5자리의 소수를 가집니다. 예를들어, 0.1234567891라면 0.12345 와 0.0000067891... 로 나눕니다. 소수부가 10자리가 넘어간다면, 소수부 뒷자리는 모두 **뒤 유전자**로 귀속됩니다. (이후 앞, 뒤 유전자는 각각 a, b라 서술합니다.)

**2) bias 유전자**

bias 는 각각 개체가 본연의 값을 가집니다. 해당 개체의 bias 본연의 값을 유전자로 가집니다.

**교배방식**

------

교배방식은 다음과 같습니다. population number를 90으로 설정하고, 각각 개체마다 모든 개체와 교배할 수 있도록 합니다. 예를 들어, 개체가 3마리라면 (1,2) (1,3) (2,3) 이렇게 3가지 경우가 나옵니다.

따라서 Combination을 이용해서 , selection number를 10으로 설정한다면, 10C2 = 45로 Combination하는 방법을 두번 고려한다면, 해당 population number의 다음 세대의 개체수를 얻어낼 수 있습니다.

앞에서 든 예시를 살펴보며 제 프로젝트에 교배방식을 설명하겠습니다. 예시로 (1,2), (1,3) , (2,3) 이라했는데, 앞에 있는 개체를 **‘부’**, 뒤에있는 개체를 **‘모’**라고 칭하겠습니다.

상위 개체를 포함해서, 소수부 뒷 5자리는 분류하는 평면의 이동에 드라마틱한 영향을 주지 않습니다. 따라서, 소수부 앞자리를 전달하기 위해 고민을 한 결과, 다음 두 방법으로 교배했습니다.

어떻게 보면 비트를 섞는 것도 좋은 방법이긴 하겠지만, 좋은 부모끼리의 형질이 균등하게 분배되면 더 좋은 세대를 낳지 않을까의 생각해서 다음과 같이 생각했습니다.

방법 1)

가중치 : 부의 앞자리 유전자 + 모의 뒷자리 유전자

bias : 부모의 평균 값

방법 2)

가중치 : 부모의 앞자리 유전자의 평균값

bias : 부모의 평균 값

### Mutation 규칙

------

다음 개체로의 다양성을 부여하기 위해 변이가 필요합니다. 따라서 개체당 일어날 확률은 1프로로 지정하였습니다.

코드는 쉽게 설명하면, 100개의 공이 상자에 담겨있는데, 하나의 공만 “변이”라고 적어져 있습니다. 이제 한 개체씩 돌아가며 복원추출을 해, “변이”라는 공을 뽑으면 변이가 일어납니다.

```python
for i in range(population_number):
        randomidx = random.randrange(0, 100)
        if randomidx==1:
            print("mutation occur") # mutation이 발생할 경우 알려주는 문구.
            # 변이 규칙에 의해 개체에 변이가 일어난다.
            w1_population[i] = w1_population[i] + 0.2
            w2_population[i] = w2_population[i] - 0.2
            w3_population[i] = w3_population[i] / 2
            b_population[i] = b_population[i] - 2
```

변이 시, 다음과 같은 과정을 수행합니다. `w1 = w1 + 0.2`

```
w2 = w2 - 0.2
w3 = w3 / 2
b = b - 2
```

### Replace 규칙

------

교배 후, 나온 개체들을 다음 세대로 취급합니다. 공교롭게도, 선택된 개체들이 10마리 이고, 교배 방식으로 생성되는 자식들이 2*10C2 = 90 이므로 세대 수를 유지하면서 다음 세대로 진행할 수 있습니다.

이제, 알고리즘 종료 조건에 다다를 때 까지 selection - crossover - mutation - replace 를 반복합니다.

### 종료 조건

------

Generic Algorithm을 종료하는 조건은, 주어진 목표에 다다랐는지 확인하면 됩니다.

종료조건은 , 다음 세대에게 전달 될 개체중 가장 fitness가 낮은 10번째 개체의 fitness score를 검사합니다.

fitness score를 검사해, 값이 0.0001 보다 작으면, 적합한 model로 판단합니다.

![1-5output.JPG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e78cf024-d13e-46be-819a-c12e570d713f/1-5output.jpg)

![1-5fig.JPG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/aa138778-aa17-4929-bd54-879de5c5c460/1-5fig.jpg)