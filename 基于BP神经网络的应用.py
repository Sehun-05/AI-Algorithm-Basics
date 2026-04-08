#基于bp神经网络的应用

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# 1. 加载并预处理数据
def prepare_data():
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data  # 特征：4个维度（花萼长度、宽度，花瓣长度、宽度）
    y = iris.target  # 标签：0,1,2（对应3种鸢尾花）

    # 特征标准化（标准化）：使每个特征均值为0，标准差为1，加速训练
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 标签独热编码：将0,1,2转为one-hot向量（如0→[1,0,0]，1→[0,1,0]）
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # 划分训练集（80%）和测试集（20%）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, iris.target_names


# 2. 定义BP神经网络类
class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化网络参数
        input_size: 输入层神经元数（特征数）
        hidden_size: 隐藏层神经元数
        output_size: 输出层神经元数（类别数）
        learning_rate: 学习率
        """
        # 初始化权重（随机值）和偏置（0）
        self.weights1 = np.random.randn(input_size, hidden_size)  # 输入层→隐藏层权重
        self.bias1 = np.zeros((1, hidden_size))  # 隐藏层偏置
        self.weights2 = np.random.randn(hidden_size, output_size)  # 隐藏层→输出层权重
        self.bias2 = np.zeros((1, output_size))  # 输出层偏置
        self.learning_rate = learning_rate
        self.loss_history = []  # 记录训练过程中的损失值

    def sigmoid(self, x):
        """激活函数：sigmoid（将值映射到0-1之间）"""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """sigmoid的导数（用于反向传播）"""
        return x * (1 - x)

    def forward(self, X):
        """前向传播：计算网络输出"""
        # 隐藏层计算：输入→加权求和→加偏置→激活
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        # 输出层计算：隐藏层输出→加权求和→加偏置→激活
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, output):
        """反向传播：计算梯度并更新权重和偏置"""
        # 计算输出层误差（真实值-预测值）
        output_error = y - output
        # 输出层梯度 = 误差 × 激活函数导数
        output_delta = output_error * self.sigmoid_derivative(output)

        # 计算隐藏层误差（输出层误差通过权重反向传播）
        hidden_error = output_delta.dot(self.weights2.T)
        # 隐藏层梯度 = 误差 × 激活函数导数
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        # 更新权重和偏置（梯度下降）
        self.weights2 += self.hidden.T.dot(output_delta) * self.learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights1 += X.T.dot(hidden_delta) * self.learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        """训练网络"""
        for epoch in range(epochs):
            # 前向传播得到预测值
            output = self.forward(X)
            # 计算损失（均方误差）
            loss = np.mean(np.square(y - output))
            self.loss_history.append(loss)
            # 反向传播更新参数
            self.backward(X, y, output)
            # 每1000轮打印一次损失
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """预测：返回概率最大的类别索引"""
        output = self.forward(X)
        return np.argmax(output, axis=1)


# 3. 主函数：训练并评估模型
def main():
    # 准备数据
    X_train, X_test, y_train, y_test, class_names = prepare_data()
    input_size = X_train.shape[1]  # 输入层大小：4（特征数）
    hidden_size = 8  # 隐藏层大小：8（可调整）
    output_size = y_train.shape[1]  # 输出层大小：3（类别数）

    # 创建并训练BP神经网络
    model = BPNeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)
    model.train(X_train, y_train, epochs=15000)  # 训练15000轮

    # 在测试集上评估
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)  # 将独热编码转回类别索引
    accuracy = np.mean(y_pred == y_true) * 100
    print(f"\n测试集准确率：{accuracy:.2f}%")

    # 打印部分预测结果
    print("\n部分预测结果（真实值 → 预测值）：")
    for i in range(10):  # 打印前10个测试样本
        print(f"{class_names[y_true[i]]} → {class_names[y_pred[i]]}")

    # 绘制损失曲线
    plt.plot(model.loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.show()


if __name__ == "__main__":
    main()