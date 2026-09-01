哈囉，親愛的程式學習者！恭喜你，我們又一起邁向了 MLOps 旅程的一個重要里程碑！到了【第 121 天】，你已經不再只是個程式碼的撰寫者，更是一位模型生命週期的管理者了。今天，我們要探討一個超級實用的主題：**資料漂移 (Data Drift) 檢測與模型再訓練 (Model Retraining)**。

想像一下，你辛辛苦苦訓練出了一個超棒的預測模型，它在測試時表現得無懈可擊。你興奮地將它部署到真實世界中。一開始，一切都很順利，但幾週、幾個月過去了，你開始發現模型的表現怎麼越來越差？這時候，你可能就遇到了「資料漂移」這個潛在的元兇！

---

### ✨ **什麼是資料漂移 (Data Drift)？**

簡單來說，資料漂移就是指**模型訓練時所使用的資料特徵分佈，與模型在實際運行時遇到的新資料特徵分佈之間產生了顯著的變化**。

你可以這樣想：你的模型就像一位經驗豐富的氣象預報員。他根據過去 20 年的氣象資料學會了預測天氣。但如果突然全球氣候變遷，天氣模式與過去完全不同了，這位預報員的預測準確度自然會大幅下降。這就是資料漂移！

**為什麼會發生？**

*   **世界變了：** 市場趨勢、使用者行為、產品特性、傳感器故障、新政策等等。
*   **資料收集方式變了：** 新的資料來源、資料處理流程調整。

**為什麼重要？**

資料漂移會直接導致你的模型表現下降，甚至做出錯誤的預測，讓使用者體驗變差，甚至造成業務損失！所以，**學會偵測並處理資料漂移，是 MLOps 中至關重要的一環。**

---

### 🕵️‍♀️ **如何偵測資料漂移？**

偵測資料漂移的方法有很多種，從簡單的統計檢定到複雜的監控系統。今天，我們要介紹一個超級好用的 Python 套件：`Evidently AI`。它能幫助我們快速生成視覺化報告，一目了然地看到資料哪裡發生了變化。

首先，請確保你安裝了 `evidently`：
```bash
pip install evidently scikit-learn pandas numpy
```

接下來，讓我們用程式碼來模擬一些資料漂移吧！

```python
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("--- 模擬資料與漂移偵測 ---")

# 1. 準備訓練資料 (Reference Data)
np.random.seed(42)
n_samples = 1000
train_data = {
    'feature_1': np.random.normal(loc=10, scale=2, size=n_samples), # 特徵 1
    'feature_2': np.random.normal(loc=5, scale=1, size=n_samples),  # 特徵 2
    'target': np.random.normal(loc=100, scale=10, size=n_samples) + 
              np.random.normal(loc=10, scale=2, size=n_samples) * 5 # 目標值與特徵 1 強相關
}
train_df = pd.DataFrame(train_data)
print(f"訓練資料 (train_df) 前 5 筆:\n{train_df.head()}\n")

# 2. 模擬線上新資料 (Current Data)，加入資料漂移
# 假設 feature_1 的平均值變高了，且 target 的分佈也輕微改變
current_data = {
    'feature_1': np.random.normal(loc=12, scale=2, size=n_samples), # feature_1 平均值從 10 漂移到 12
    'feature_2': np.random.normal(loc=5.1, scale=1, size=n_samples), # feature_2 輕微漂移
    'target': np.random.normal(loc=110, scale=12, size=n_samples) + # 目標值整體升高
              np.random.normal(loc=12, scale=2, size=n_samples) * 5
}
current_df = pd.DataFrame(current_data)
print(f"線上新資料 (current_df) 前 5 筆 (已漂移):\n{current_df.head()}\n")

# 3. 使用 Evidently AI 偵測資料漂移
print("開始使用 Evidently AI 偵測資料漂移...")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

# 執行報告，比較 current_df 和 train_df
data_drift_report.run(current_data=current_df, reference_data=train_df, column_mapping=None)

# 顯示報告 (會在你的程式執行目錄下生成一個 HTML 檔案，或在 Jupyter Notebook 中直接顯示)
# data_drift_report.show() 
# 你也可以將報告儲存為 HTML 檔案方便查看
report_path = "data_drift_report.html"
data_drift_report.save_html(report_path)
print(f"資料漂移報告已儲存至: {report_path}")
print("請打開這個 HTML 檔案，你會看到詳細的漂移分析！")

# 檢查報告摘要判斷是否有漂移
if data_drift_report.as_dict()['metrics'][0]['result']['dataset_drift']:
    print("\n🚨 警報！Evidently AI 報告顯示偵測到顯著的資料漂移！")
    for feature, drift_info in data_drift_report.as_dict()['metrics'][0]['result']['features'].items():
        if drift_info['drift_detected']:
            print(f"   - 特徵 '{feature}' 偵測到漂移！")
else:
    print("\n✅ Evidently AI 報告顯示目前資料沒有顯著漂移。")

print("\n----------------------------------\n")
```

執行上面的程式碼後，你會在專案目錄中看到一個 `data_drift_report.html` 檔案。打開它！你將會看到一個豐富的互動式儀表板，詳細說明了每個特徵的分佈如何從 `train_df` 漂移到 `current_df`，以及整體資料集的漂移情況。你會發現 `feature_1` 顯示有明顯的漂移！

---

### 🔄 **偵測到漂移後怎麼辦？模型再訓練！**

當我們確認資料發生了漂移，並且它已經影響到模型的預測效能時，最直接且有效的方法就是**模型再訓練**。

**什麼是模型再訓練？**

模型再訓練就是用新的、更能夠代表目前資料分佈的資料集，重新訓練你的模型。這就像是那位氣象預報員，在氣候變遷後，學習了最新的氣象數據，更新了他的預報知識。

**為什麼要再訓練？**

*   **適應變化：** 讓模型學習到新的資料模式和特徵分佈。
*   **恢復效能：** 透過更新模型，使其預測準確度回到可接受的水平。

讓我們來看一個簡單的再訓練範例。

```python
print("--- 模型訓練與再訓練 ---")

# 1. 初始模型訓練
print("1. 使用初始訓練資料訓練模型...")
X_train_initial = train_df[['feature_1', 'feature_2']]
y_train_initial = train_df['target']

model_initial = LinearRegression()
model_initial.fit(X_train_initial, y_train_initial)

# 模擬初始模型在線上資料上的表現 (預期會變差)
y_pred_initial_on_current = model_initial.predict(current_df[['feature_1', 'feature_2']])
mae_initial_on_current = mean_absolute_error(current_df['target'], y_pred_initial_on_current)
print(f"初始模型在漂移後的資料上 MAE: {mae_initial_on_current:.2f}")

# 2. 模型再訓練
# 假設我們現在收集到了新的、代表當前真實世界分佈的資料 (這裡我們直接用 current_df 來代表)
print("\n2. 使用新的資料進行模型再訓練...")
X_retrain = current_df[['feature_1', 'feature_2']]
y_retrain = current_df['target']

model_retrained = LinearRegression()
model_retrained.fit(X_retrain, y_retrain)

# 評估再訓練模型在新的線上資料上的表現 (預期會改善)
y_pred_retrained_on_current = model_retrained.predict(current_df[['feature_1', 'feature_2']])
mae_retrained_on_current = mean_absolute_error(current_df['target'], y_pred_retrained_on_current)
print(f"再訓練模型在漂移後的資料上 MAE: {mae_retrained_on_current:.2f}")

# 比較兩種模型的效能
if mae_retrained_on_current < mae_initial_on_current:
    print("\n🎉 恭喜！模型再訓練成功改善了在漂移資料上的效能！")
else:
    print("\n🤔 再訓練後效能似乎沒有改善，可能需要更深入的分析或更大量的更新資料。")

print("\n----------------------------------\n")
```

你會看到，經過再訓練的模型在 `current_df` 上的 `MAE` (平均絕對誤差) 會比初始模型在 `current_df` 上的 `MAE` 小得多。這證明了再訓練的有效性！

---

### 🚀 **MLOps 的自動化循環**

在真實的 MLOps 環境中，這個過程通常會被自動化：

1.  **資料監控：** 持續收集新的線上資料。
2.  **漂移檢測：** 定期或當資料量達到一定閾值時，自動執行像 `Evidently AI` 這樣的工具來檢測資料漂移。
3.  **警報機制：** 一旦偵測到顯著漂移，自動發出警報給相關團隊。
4.  **自動再訓練：** 在符合特定條件時（例如漂移程度超過閾值，或模型效能下降到某個程度），觸發自動化的模型再訓練流程。
5.  **模型部署：** 訓練好的新模型會經過驗證後，自動部署上線，替換舊模型。
6.  **持續監控：** 這個循環會不斷地進行，確保模型始終保持最佳效能。

---

### 🎉 **總結與鼓勵**

哇！今天我們學到了好多東西！從了解資料漂移的危害，到親手使用 `Evidently AI` 偵測它，最後再用程式碼實現模型的再訓練。這一步步的實踐，讓你對 MLOps 的核心概念又有了更深的理解。

記住，部署模型並不是終點，而是另一個旅程的開始。你的模型是活的，它會隨著世界而變化，而你的任務，就是像一位盡責的園丁，細心照料它，讓它能持續茁壯成長。

你已經邁出了重要的一步！在 MLOps 的世界裡，這是一個至關重要的技能。繼續保持好奇心，不斷探索，你的程式之路會越走越寬廣！我們下一個挑戰再見！