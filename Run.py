import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from google.colab import files

# 删除之前保存的文件
file_path = 'data.xlsx'
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Deleted existing file: {file_path}")

# 上传Excel文件
uploaded = files.upload()

# 读取上传的Excel文件
# 上传的文件名为 'data.xlsx'
if not uploaded:
    print("No file was uploaded. Please try again.")
else:
    # 上传的文件名为 'data.xlsx'
    file_name = list(uploaded.keys())[0]
    if not file_name.endswith('.xlsx'):
        print(f"Uploaded file is not an Excel file. Please upload a file with '.xlsx' extension.")
    else:
        # 读取上传的Excel文件
        df = pd.read_excel(file_name)

# 加载数据（确保这里是实际想要建模的数据列）
days = df.iloc[:, 1].values         # 投放天数
budget = df.iloc[:, 2].values       # 日均预算
cvr = df.iloc[:, 3].values          # CVR
roi = df.iloc[:, 4].values          # ROI

# 标准化预算、天数和CVR（避免绝对数值差异过大导致模型权重失衡）
scaler_budget = StandardScaler()
scaler_days = StandardScaler()
scaler_cvr = StandardScaler()

budget_scaled = scaler_budget.fit_transform(budget.reshape(-1, 1))  # 标准化预算
days_scaled = scaler_days.fit_transform(days.reshape(-1, 1))        # 标准化天数
cvr_scaled = scaler_cvr.fit_transform(cvr.reshape(-1, 1))           # 标准化CVR

# 创建输入数据矩阵 X，包括预算、天数和CVR，
X = np.hstack([budget_scaled, days_scaled, cvr_scaled])

# 使用 ROI 作为目标变量 y
y = roi

# 划分训练集和测试集 test_size = 0.2即训练数据的比例为20%，random_state=42 这里设置一个随机的种子，确保每次实验的结果相同便于对比
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林回归模型 n_estimators=100，这个随机森林由100棵决策树组成（即弱学习器），避免单个学习器的过拟合问题
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用网格搜索进行超参数调优，param_grid 相当于一个字典，定义了要搜索的超参数的候选值。
param_grid = {
    'n_estimators': [50, 100, 200],  #决策树数量
    'max_depth': [None, 5, 10], #决策树的最大深度
    'min_samples_split': [2, 5, 10] #节点再划分的最小样本数
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 最优模型
best_rf = grid_search.best_estimator_

# 预测并计算误差
y_pred = best_rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}") #计算模型方差，判断模型的拟合度

# ROI的全局计算
def compute_weighted_average_optimized():
    total_weight = 0
    weighted_budget_sum = 0
    weighted_days_sum = 0
    overall_max_predicted_roi = -np.inf  # 用于存储全局的最大预测 ROI

    for i in range(len(budget_scaled)):
        max_predicted_roi = -np.inf
        best_budget = None
        best_days = None

        # 减少搜索点数量
        for budget_value in np.linspace(budget_scaled.min(), budget_scaled.max(), 20):  # 从100减少到20
            for days_value in np.linspace(days_scaled.min(), days_scaled.max(), 20):    # 从100减少到20
                input_features = np.hstack((
                    np.array([[budget_value]]),
                    np.array([[days_value]]),
                    np.array([[cvr_scaled[i][0]]])
                ))

                predicted_roi = best_rf.predict(input_features)[0]

                if predicted_roi > max_predicted_roi:
                    max_predicted_roi = predicted_roi
                    best_budget = budget_value
                    best_days = days_value

        # 将标准化的预算和天数反标准化（因为前面做了归一化处理，为了可读，输出结果还是还原成原始数据）
        best_budget = scaler_budget.inverse_transform([[best_budget]])[0][0]
        best_days = scaler_days.inverse_transform([[best_days]])[0][0]

        # 使用最大预测ROI作为权重
        total_weight += max_predicted_roi
        weighted_budget_sum += best_budget * max_predicted_roi
        weighted_days_sum += best_days * max_predicted_roi

        # 更新全局的最大预测 ROI
        if max_predicted_roi > overall_max_predicted_roi:
            overall_max_predicted_roi = max_predicted_roi

    # 计算加权平均预算和天数
    weighted_avg_budget = weighted_budget_sum / total_weight
    weighted_avg_days = weighted_days_sum / total_weight

    return weighted_avg_budget, weighted_avg_days, overall_max_predicted_roi

# 计算全局的加权平均预算和天数
avg_budget, avg_days, max_predicted_roi = compute_weighted_average_optimized()
print(f"Weighted Average Budget: {avg_budget}")
print(f"Weighted Average Days: {avg_days}")
print(f"Max Predicted ROI: {max_predicted_roi}")

# 对所有数据点进行预测，比较实际 ROI 和预测 ROI
all_predicted_roi = best_rf.predict(X)

# 将实际 ROI 与预测值输出为 DataFrame 进行比较
results = pd.DataFrame({
    'Actual ROI': y,
    'Predicted ROI': all_predicted_roi
})

# 输出每个数据点的预测值
print("\nPredicted ROI for each data point:")
print(results)

# 保存结果到 Excel 文件
results.to_excel('predicted_vs_actual_roi.xlsx', index=False)

# 使用 Colab 文件工具来下载文件
from google.colab import files
files.download('predicted_vs_actual_roi.xlsx')

#@title 模型训练 Option2 随机森林回归 (Random Forest Regression) - 4

import matplotlib.pyplot as plt

def plot_roi_changes_fixed_budget(budget=800000, days_range=(1, 14)): #日均预算80万，投放天数1-14
    """
    绘制固定预算条件下，不同投放天数下的 ROI 变化。
    
    :param budget: 固定的日均预算
    :param days_range: 投放天数的范围
    """
    # 将预算标准化
    budget_scaled = scaler_budget.transform([[budget]])[0][0]
    
    rois = []
    days = list(range(days_range[0], days_range[1] + 1))
    
    for day in days:
        # 将天数标准化
        days_scaled = scaler_days.transform([[day]])[0][0]
        
        # 创建输入特征
        input_features = np.hstack((np.array([[budget_scaled]]), np.array([[days_scaled]]), np.array([[cvr_scaled.mean()]])))
        
        # 预测 ROI
        predicted_roi = best_rf.predict(input_features)[0]
        rois.append(predicted_roi)
    
    # 绘制 ROI 变化图
    plt.figure(figsize=(10, 6))
    plt.plot(days, rois, marker='o')
    plt.title(f"ROI Changes with Fixed Budget of {budget} and Varying Days")
    plt.xlabel("Days")
    plt.ylabel("Predicted ROI")
    plt.grid(True)
    plt.show()

# 调用函数
plot_roi_changes_fixed_budget(budget=800000, days_range=(1, 14))

def plot_roi_changes_fixed_days(days=6, budget_range=(100000, 1000000, 100000)):
    """
    绘制固定投放天数条件下，不同日均预算下的 ROI 变化。
    
    :param days: 固定的投放天数
    :param budget_range: 日均预算的范围和步长 (start, stop, step)
    """
    # 将天数标准化
    days_scaled = scaler_days.transform([[days]])[0][0]
    
    rois = []
    budgets = list(range(budget_range[0], budget_range[1] + 1, budget_range[2]))
    
    for budget in budgets:
        # 将预算标准化
        budget_scaled = scaler_budget.transform([[budget]])[0][0]
        
        # 创建输入特征
        input_features = np.hstack((np.array([[budget_scaled]]), np.array([[days_scaled]]), np.array([[cvr_scaled.mean()]])))
        
        # 预测 ROI
        predicted_roi = best_rf.predict(input_features)[0]
        rois.append(predicted_roi)
    
    # 绘制 ROI 变化图
    plt.figure(figsize=(10, 6))
    plt.plot(budgets, rois, marker='o')
    plt.title(f"ROI Changes with Fixed Days of {days} and Varying Budgets")
    plt.xlabel("Budget (in 10,000)")
    plt.ylabel("Predicted ROI")
    plt.grid(True)
    plt.show()

# 调用函数
plot_roi_changes_fixed_days(days=6, budget_range=(100000, 1000000, 100000))