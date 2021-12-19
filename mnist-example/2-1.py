import numpy as np
import pandas as pd

# 각각 1, 2, 3 이 200개씩 있고, 한 글자당 784개의 픽셀을 가진다.
total_number_count = 600
pixel_per_word = 784

training_data = pd.read_csv("p2_training_data_mnist.csv")   # 실제 손글씨 데이터
training_data_label = pd.read_csv("p2_training_data_label.csv") # 실제 데이터의 레이블

np_training_data = np.array(training_data) # numpy.array로 변경 784개의 특성 값을 가진다.


