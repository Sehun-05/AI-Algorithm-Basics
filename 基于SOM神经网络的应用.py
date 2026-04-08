import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.stats import mode


# 1. 数据加载与预处理优化
def load_data(use_selected_features=True):
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # 特征选择：优先使用区分度高的花瓣特征（花瓣长度、花瓣宽度）
    if use_selected_features:
        X = X[:, 2:]  # 保留后两个特征（花瓣特征）

    # 使用MinMaxScaler归一化到[0,1]，更适合SOM权重范围
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, class_names, X  # 返回原始特征用于后续分析


# 2. 改进的SOM神经网络类
class ImprovedSOM:
    def __init__(self, input_dim, output_size=(15, 15), learning_rate=0.05, sigma=2.0):
        """
        改进点：
        - 缩小输出网格（15x15）适合小样本
        - 优化初始学习率和邻域半径
        """
        self.input_dim = input_dim
        self.output_rows, self.output_cols = output_size
        self.learning_rate = learning_rate
        self.sigma = sigma

        # 权重初始化：从输入样本中随机选择（而非纯随机）
        self.weights = None  # 后续在train中初始化

        # 生成网格坐标
        self.rows = np.arange(self.output_rows)
        self.cols = np.arange(self.output_cols)
        self.grid_x, self.grid_y = np.meshgrid(self.rows, self.cols, indexing='ij')

        # 记录训练过程中的误差
        self.error_history = []

    def find_bmu(self, x):
        """找到最佳匹配单元（BMU）"""
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_weights(self, x, bmu, epoch, max_epochs):
        """改进的权重更新：线性衰减学习率和邻域半径"""
        # 线性衰减（比指数衰减更稳定）
        lr = self.learning_rate * (1 - epoch / max_epochs)
        sigma = self.sigma * (1 - epoch / max_epochs)

        bmu_row, bmu_col = bmu

        # 计算邻域距离
        distance = np.sqrt((self.grid_x - bmu_row) ** 2 + (self.grid_y - bmu_col) ** 2)
        # 高斯邻域函数
        neighborhood = np.exp(-(distance ** 2) / (2 * sigma **2))

        # 更新权重
        self.weights += lr * neighborhood[..., np.newaxis] * (x - self.weights)

    def train(self, X, epochs=20000, batch_size=8):
        """
        改进点：
        - 增加训练轮次（20000次）
        - 小批次更新（增强稳定性）
        - 从样本中初始化权重
        """
        # 从输入样本中初始化权重（加速收敛）
        if self.weights is None:
            n_weights = self.output_rows * self.output_cols
            self.weights = X[np.random.choice(len(X), size=n_weights, replace=True)]
            self.weights = self.weights.reshape(self.output_rows, self.output_cols, self.input_dim)

        for epoch in range(epochs):
            # 小批次更新（每次随机选batch_size个样本）
            batch_indices = np.random.choice(len(X), size=batch_size, replace=False)
            batch = X[batch_indices]

            # 累积权重更新量
            weight_update = np.zeros_like(self.weights)

            for x in batch:
                bmu = self.find_bmu(x)
                bmu_row, bmu_col = bmu
                lr = self.learning_rate * (1 - epoch / epochs)
                sigma = self.sigma * (1 - epoch / epochs)

                # 计算邻域
                distance = np.sqrt((self.grid_x - bmu_row)** 2 + (self.grid_y - bmu_col) **2)
                neighborhood = np.exp(-(distance** 2) / (2 * sigma **2))

                # 累加更新量
                weight_update += lr * neighborhood[..., np.newaxis] * (x - self.weights)

            # 应用批次平均更新
            self.weights += weight_update / batch_size

            # 记录训练误差（每1000轮计算一次）
            if epoch % 1000 == 0:
                total_error = np.mean([np.linalg.norm(x - self.weights[self.find_bmu(x)]) for x in X])
                self.error_history.append(total_error)
                print(f"Epoch {epoch}/{epochs}, Reconstruction Error: {total_error:.6f}")

    def map_samples(self, X):
        """映射样本到SOM网格"""
        return np.array([self.find_bmu(x) for x in X])


# 3. 后处理与评估函数
def post_process(bmu_coords, true_labels, n_clusters=3):
    """
    改进点：
    - 使用KMeans对BMU坐标二次聚类
    - 计算准确率和调整兰德指数（ARI）
    """
    # 对BMU坐标进行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    som_labels = kmeans.fit_predict(bmu_coords)

    # 映射聚类标签到真实标签（解决类别编号不匹配问题）
    labeled_clusters = np.zeros_like(som_labels)
    for cluster in range(n_clusters):
        mask = (som_labels == cluster)
        labeled_clusters[mask] = mode(true_labels[mask])[0]

    # 计算评估指标
    accuracy = accuracy_score(true_labels, labeled_clusters)
    ari = adjusted_rand_score(true_labels, som_labels)  # 衡量聚类与真实标签的一致性
    return labeled_clusters, accuracy, ari


# 4. 可视化函数（增强版）
def visualize_results(bmu_coords, true_labels, pred_labels, class_names, som):
    """
    改进点：
    - 同时显示真实标签和预测标签
    - 增加聚类边界
    - 展示训练误差曲线
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    markers = ['o', 's', 'D']
    colors = ['r', 'g', 'b']

    # 1. 真实标签分布
    ax1 = axes[0]
    for i in range(len(bmu_coords)):
        row, col = bmu_coords[i]
        ax1.scatter(col, row, color=colors[true_labels[i]], marker=markers[true_labels[i]],
                    alpha=0.7, label=class_names[true_labels[i]] if i < 3 else "")
    ax1.set_title('True Labels on SOM Grid')
    ax1.set_xlabel('SOM Column')
    ax1.set_ylabel('SOM Row')
    ax1.legend()

    # 2. 预测标签分布
    ax2 = axes[1]
    for i in range(len(bmu_coords)):
        row, col = bmu_coords[i]
        ax2.scatter(col, row, color=colors[pred_labels[i]], marker=markers[pred_labels[i]],
                    alpha=0.7, label=class_names[pred_labels[i]] if i < 3 else "")
    ax2.set_title('Predicted Labels on SOM Grid')
    ax2.set_xlabel('SOM Column')
    ax2.set_ylabel('SOM Row')
    ax2.legend()

    # 3. 训练误差曲线
    ax3 = axes[2]
    ax3.plot(range(0, len(som.error_history)*1000, 1000), som.error_history)
    ax3.set_title('Training Reconstruction Error')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Distance to BMU')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# 5. 主函数
def main():
    # 加载数据（使用筛选后的特征）
    X_scaled, y, class_names, X_original = load_data(use_selected_features=True)
    input_dim = X_scaled.shape[1]  # 筛选后为2维（花瓣特征）

    # 创建改进的SOM模型
    som = ImprovedSOM(
        input_dim=input_dim,
        output_size=(15, 15),  # 缩小网格
        learning_rate=0.05,
        sigma=2.0
    )

    # 训练模型
    som.train(X_scaled, epochs=20000, batch_size=8)

    # 映射样本到SOM网格
    bmu_coords = som.map_samples(X_scaled)

    # 后处理与评估
    pred_labels, accuracy, ari = post_process(bmu_coords, y)
    print(f"\n分类准确率: {accuracy:.2%}")
    print(f"调整兰德指数(ARI): {ari:.4f} (越接近1越好)")

    # 可视化结果
    visualize_results(bmu_coords, y, pred_labels, class_names, som)


if __name__ == "__main__":
    main()