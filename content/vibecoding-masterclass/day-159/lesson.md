哈囉，程式探險家們！🎉 歡迎來到你的【第 159 天】MLOps 學習之旅！

你已經學會如何訓練模型，這很棒！但你有沒有想過，當你的模型部署上線後，它會一直表現良好嗎？現實世界是多變的，數據會隨著時間改變，模型的表現也會隨之下降。這時候，我們就需要 MLOps 的超能力出場了：**模型監控與效能管理**！

想像一下你的汽車，如果引擎燈亮了，你會怎麼辦？你會趕快檢查是不是哪裡出了問題。我們的機器學習模型也需要這樣的「引擎燈」—— 這就是模型監控。它能幫助我們在問題變大之前，就發現並解決它們。

今天，我們就要來實戰演練，學習如何為你的模型搭建一個監控系統，確保它在生產環境中始終保持最佳狀態！

---

### 一、模型監控 (Model Monitoring)：模型的健康檢查官

模型監控就像是模型的健康檢查官，它持續觀察模型在生產環境中的表現，並在出現異常時發出警報。主要監控的面向包括：

1.  **數據漂移 (Data Drift)**：
    *   **特徵漂移 (Feature Drift)**：輸入資料的統計特性（例如分佈、平均值）隨時間發生變化。
    *   **標籤漂移 (Label Drift)**：目標變數 (target variable) 的分佈發生變化。
    *   **概念漂移 (Concept Drift)**：輸入特徵與目標變數之間的關係發生變化，例如客戶行為模式改變。
2.  **模型效能 (Model Performance)**：
    *   模型在實際生產數據上的預測準確度、精確度、召回率、F1-score (分類模型) 或 MSE、RMSE (迴歸模型) 等指標是否下降。
3.  **系統資源 (System Metrics)**：
    *   模型的推理延遲、請求成功率、CPU/GPU 使用率等。

當監控系統偵測到上述任何一種「漂移」或效能下降時，就能即時通知我們，讓我們可以介入處理。

---

### 二、實戰：使用 `Evidently AI` 進行模型監控

市面上有許多 MLOps 工具可以幫助我們進行模型監控，例如 MLflow、Arize、Whylogs 等。今天，我們將使用一個輕量級且強大的開源工具：[`Evidently AI`](https://evidentlyai.com/)。它能快速生成詳細的數據和模型效能報告。

首先，確保你已經安裝了必要的套件：

```bash
pip install pandas scikit-learn evidently
```

接著，讓我們模擬一個情境：你訓練了一個簡單的分類模型，並將它部署上線。一段時間後，你收集了一些新的生產數據，想要檢查模型和數據的健康狀況。

#### 1. 準備模擬數據與模型

我們將建立兩份數據：一份是模型訓練時的「參考數據」(reference data)，另一份是模型上線一段時間後收集到的「當前數據」(current data)。

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPerformancePreset

# --- 1. 建立模擬數據 ---
np.random.seed(42)

# 參考數據 (訓練時的數據)
n_ref = 1000
reference_data = pd.DataFrame({
    'feature_1': np.random.rand(n_ref) * 10,
    'feature_2': np.random.normal(5, 2, n_ref),
    'feature_3': np.random.randint(0, 5, n_ref),
    'target': np.random.randint(0, 2, n_ref) # 二元分類目標
})

# 模擬一個簡單的模型預測
model = RandomForestClassifier(random_state=42)
X_ref = reference_data[['feature_1', 'feature_2', 'feature_3']]
y_ref = reference_data['target']
model.fit(X_ref, y_ref)

reference_data['prediction'] = model.predict(X_ref)
reference_data['prediction_proba_0'] = model.predict_proba(X_ref)[:, 0]
reference_data['prediction_proba_1'] = model.predict_proba(X_ref)[:, 1]

# 當前數據 (模擬生產環境中收集到的數據，略有漂移)
n_curr = 500
current_data = pd.DataFrame({
    'feature_1': np.random.rand(n_curr) * 12, # feature_1 輕微漂移
    'feature_2': np.random.normal(5.5, 2.5, n_curr), # feature_2 輕微漂移
    'feature_3': np.random.randint(0, 6, n_curr), # feature_3 輕微漂移，多一個類別
    'target': np.random.randint(0, 2, n_curr) # 二元分類目標 (實際生產數據的真實標籤)
})

# 用已訓練的模型對當前數據進行預測
X_curr = current_data[['feature_1', 'feature_2', 'feature_3']]
current_data['prediction'] = model.predict(X_curr)
current_data['prediction_proba_0'] = model.predict_proba(X_curr)[:, 0]
current_data['prediction_proba_1'] = model.predict_proba(X_curr)[:, 1]

print("數據準備完成！")
print("參考數據前 5 行:\n", reference_data.head())
print("\n當前數據前 5 行:\n", current_data.head())
```

#### 2. 數據漂移監控 (Data Drift Monitoring)

現在，我們來看看 `Evidently` 如何幫助我們檢測數據漂移。

```python
# --- 2. 數據漂移監控 ---
print("\n--- 執行數據漂移報告 ---")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

data_drift_report.run(reference_data=reference_data, current_data=current_data)

# 顯示報告 (會在Jupyter Notebook或支援HTML顯示的環境中直接顯示)
# 你也可以將報告儲存為HTML檔案
# data_drift_report.save_html("data_drift_report.html")
# print("數據漂移報告已儲存為 data_drift_report.html")
data_drift_report.show()
```

運行上面的程式碼，你會看到一個詳細的 HTML 報告。其中會清楚標示出哪些特徵發生了統計上的顯著變化（用紅色表示漂移，橙色表示潛在漂移）。你會看到 `feature_1`, `feature_2`, `feature_3` 應該會有不同程度的漂移警示，因為我們在生成 `current_data` 時故意引入了變化。

#### 3. 模型效能監控 (Model Performance Monitoring)

接下來，我們檢查模型的實際效能。對於分類模型，我們會關注準確度、精確度、召回率、F1 分數等。

```python
# --- 3. 模型效能監控 ---
print("\n--- 執行模型效能報告 ---")
performance_report = Report(metrics=[
    ClassificationPerformancePreset(),
])

# 需要定義column_mapping，告訴Evidently哪些列是目標、預測和機率
column_mapping = {
    "target": "target",
    "prediction": "prediction",
    "prediction_probas": ["prediction_proba_0", "prediction_proba_1"], # 如果是二元分類，需要提供每個類別的機率
    "numerical_features": ['feature_1', 'feature_2'],
    "categorical_features": ['feature_3'],
}

performance_report.run(reference_data=reference_data, 
                       current_data=current_data, 
                       column_mapping=column_mapping)

# 顯示報告
# performance_report.save_html("performance_report.html")
# print("模型效能報告已儲存為 performance_report.html")
performance_report.show()
```

這個報告將比較參考數據和當前數據上的模型效能指標。如果你的模型在當前數據上的準確度、F1 分數等指標明顯下降，那麼就是時候考慮介入了！你還可以看到決策閾值、混淆矩陣、ROC 曲線等更多細節。

---

### 三、效能管理 (Performance Management)：當警報響起時

當模型監控發現問題時，這就是「效能管理」登場的時刻。你可能需要採取以下措施：

1.  **發出警報 (Alerting)**：
    *   整合到 Slack、Email 或其他監控系統，自動通知相關團隊。
2.  **根本原因分析 (Root Cause Analysis)**：
    *   深入分析漂移的數據或下降的效能指標，找出問題的具體原因。是數據源變了？還是用戶行為變了？
3.  **模型再訓練 (Model Retraining)**：
    *   使用新的、更能代表當前現實的數據集重新訓練模型。這通常是解決模型效能下降最常見的方法。
4.  **模型回滾 (Model Rollback)**：
    *   如果新模型表現不佳，或發生嚴重問題，需要快速回滾到之前表現良好的版本。
5.  **A/B 測試新模型 (A/B Testing)**：
    *   在部署新模型前，先在小部分流量上進行測試，確保其效能優於舊模型。

---

### 四、總結與展望

恭喜你，今天的實戰讓你對 MLOps 的模型監控有了更深入的認識！我們學會了如何使用 `Evidently AI` 偵測數據漂移和監控模型效能。這只是監控的冰山一角，在實際的 MLOps 環境中，你還需要考慮：

*   **自動化報告生成與排程**：定期生成報告，而不是手動執行。
*   **閾值設定與警報系統**：當某些指標超過預設閾值時，自動發出通知。
*   **與 ML Pipeline 整合**：將監控作為 MLOps 工作流的一部分。

模型監控是 MLOps 中至關重要的一環，它確保你的 AI 系統能夠在不斷變化的世界中持續提供價值。這是一個持續學習和優化的過程，但有了這些基本工具和概念，你已經邁出了堅實的一步！

繼續加油！你的 MLOps 旅程才剛剛開始！🚀