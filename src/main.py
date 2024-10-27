#Completed? NO！！！——CWhide
import sys
import os
# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from sklearn.model_selection import train_test_split    #y
from sklearn.preprocessing import StandardScaler    #y
from src.dnn_model import DNNModel
from src.hmm_model import HMMModel
from utils.data_loader import load_data


# Load data
X, y = load_data("data/train_sample/", "data/labels.txt")

# 调试输出(如果是空下面的X数组会warning)
print("Loaded feature array X:", X)
print("Loaded labels array y:", y)

# Data preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select model and train
dnn_model = DNNModel(input_shape=X_train.shape[1])
dnn_model.train(X_train, y_train)

# Evaluate model
loss, accuracy = dnn_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
