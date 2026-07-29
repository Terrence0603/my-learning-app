哈囉，各位同學！恭喜你來到 MLOps (Machine Learning Operations) 的核心戰場！今天我們要探討的主題，可以說是讓你的機器學習模型在真實世界中「長壽」的關鍵：**模型漂移 (Model Drift) 偵測與自動化再訓練**。

別擔心，聽起來有點硬核，但實際上它就像是你的模型的「健康管理中心」。當模型上線後，它不是就此一勞永逸了，環境是會變化的！交通模式會變、消費者行為會變、天氣會變，這些變化都可能讓你的模型預測越來越不準，這就是所謂的「模型漂移」。

### 什麼是模型漂移 (Model Drift)？

想像一下，你訓練了一個模型來預測明天超市某商品的銷量。這個模型是根據過去一年的銷售數據訓練的。但是，如果突然來了一個新的競爭者，或者發生了全球性的疫情，消費者的購買習慣大變，那麼你原來的模型可能就「失靈」了，它再也無法準確預測銷量。這種模型表現隨時間下降的現象，就是模型漂移。

模型漂移主要有兩種：
1.  **資料漂移 (Data Drift)**：模型的輸入資料分佈發生了變化。例如，你的顧客年齡層突然變得更年輕了，而你的模型從沒見過這種資料。
2.  **概念漂移 (Concept Drift)**：輸入資料與目標變數之間的關係發生了變化。例如，以前廣告投入和銷量是正相關，現在因為市場飽和，關係變弱了。

今天的實戰，我們將主要聚焦在**資料漂移**的偵測。

### 如何偵測模型漂移？使用 `evidently` AI！

要偵測模型漂移，最常見的方法就是持續監控模型的輸入資料，並將其與訓練模型時的「參考資料」進行比較。如果兩者之間存在顯著差異，就可能發生了漂移。

這裡，我們將使用一個非常棒的 Python 函式庫叫做 `evidently`。它能幫助我們快速生成資料品質、資料漂移、模型性能漂移等報告，而且視覺化效果非常好！

首先，請確保你已經安裝了 `evidently` 和 `scikit-learn`：
```bash
pip install evidently scikit-learn pandas numpy
```

接下來，讓我們用程式碼來模擬一下：

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("所需函式庫已載入完成！")

# 1. 模擬初始訓練資料 (Reference Data)
X_ref, y_ref = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
ref_df = pd.DataFrame(X_ref, columns=[f'feature_{i}' for i in range(10)])
ref_df['target'] = y_ref

print("模擬初始訓練資料完成。")

# 2. 模擬一段時間後的新資料 (Current Data)，我們故意製造一些漂移
X_current, y_current = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=0, random_state=87)

# 故意讓某些特徵的值分佈發生變化，模擬資料漂移
X_current[:, 0] = X_current[:, 0] * 1.5 + 2 # 改變 feature_0 的均值和分佈
X_current[:, 3] = X_current[:, 3] + 1     # 改變 feature_3 的均值

current_df = pd.DataFrame(X_current, columns=[f'feature_{i}' for i in range(10)])
current_df['target'] = y_current

print("模擬新資料 (包含漂移) 完成。")

# 3. 使用 Evidently 偵測資料漂移
print("\n正在生成資料漂移報告...")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])

# 比較參考資料和當前資料
data_drift_report.run(reference_data=ref_df, current_data=current_df)

# 將報告儲存為 HTML 檔案，你可以打開這個檔案在瀏覽器中查看詳細報告
report_filename = "data_drift_report.html"
data_drift_report.save_html(report_filename)
print(f"資料漂移報告已儲存至：{report_filename}")
print("請打開此 HTML 檔案查看詳細漂移情況，特別關注 'Data Drift' 標籤！")

# 在實際應用中，你可以解析 evidently 報告的 JSON 輸出，
# 或設定閾值來判斷是否發生了顯著漂移。
# 這裡我們為了示範自動化再訓練，假設我們已經檢測到漂移。
drift_detected = True # 假設我們從報告中判斷出漂移發生了

```

運行上面的程式碼後，你會得到一個 `data_drift_report.html` 檔案。打開它，你會看到一個非常漂亮的儀表板，它會告訴你每個特徵的統計分佈如何變化，以及整體漂移情況。你會發現 `feature_0` 和 `feature_3` 有顯著的資料漂移！

### 自動化再訓練 (Automated Retraining)

一旦我們偵測到模型漂移，接下來的步驟通常是「再訓練」模型。我們需要使用最新的、能代表當前資料分佈的數據來重新訓練模型，讓它再次適應新的環境。

在 MLOps 中，這個過程必須是自動化的。當監控系統發現漂移時，它應該自動觸發再訓練流程，甚至自動部署新模型。

```python
# 4. 定義一個簡單的模型訓練函式
def train_and_evaluate_model(X, y, model_name="新模型"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=87)
    model = LogisticRegression(random_state=87, solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[{model_name}] 模型訓練完成，測試準確度：{accuracy:.4f}")
    return model, accuracy

# 5. 模擬初始模型訓練
print("\n--- 初始模型訓練 ---")
initial_model, initial_accuracy = train_and_evaluate_model(ref_df.drop('target', axis=1), ref_df['target'], "初始模型")

# 6. 自動化再訓練邏輯
print("\n--- 自動化再訓練流程 ---")
if drift_detected:
    print("偵測到模型漂移！觸發自動化再訓練...")
    # 在實際情況中，你會使用最新的資料集進行訓練，這裡我們就用 current_df 作為新的訓練資料
    retrained_model, retrained_accuracy = train_and_evaluate_model(current_df.drop('target', axis=1), current_df['target'], "再訓練模型")
    
    # 這裡可以加入部署新模型的邏輯，例如將模型儲存到模型倉庫，並更新生產環境的模型
    # 例如：
    # import joblib
    # joblib.dump(retrained_model, "retrained_model.pkl")
    # print("再訓練模型已儲存。")

    print(f"\n再訓練完成！新的模型準確度為：{retrained_accuracy:.4f}")
    print("模型已更新，可以準備部署到生產環境。")
else:
    print("未偵測到模型漂移，模型無需再訓練。")

```

### 整合與實戰流程

將上述概念整合起來，一個 MLOps 的模型漂移偵測與自動化再訓練流程通常是這樣的：

1.  **模型部署 (Deployment)**：將你的初始模型部署到生產環境。
2.  **資料收集與監控 (Data Collection & Monitoring)**：持續收集生產環境中的輸入資料，並與模型訓練時的參考資料進行比較。
3.  **漂移偵測 (Drift Detection)**：使用像 `evidently` 這樣的工具，定期（例如每天、每週）分析新資料，偵測資料分佈是否發生了顯著漂移。
4.  **警報與觸發 (Alert & Trigger)**：如果偵測到漂移，系統會發出警報（例如發送郵件、Slack 訊息），並自動觸發再訓練流程。
5.  **自動化再訓練 (Automated Retraining)**：使用最新的、包含當前資料分佈的資料集，自動重新訓練模型。
6.  **模型評估與部署 (Evaluation & Deployment)**：評估新訓練模型的性能。如果性能滿意，自動將新模型部署到生產環境，替換舊模型。

### 總結與挑戰

今天我們學習了 MLOps 中一個至關重要的概念：模型漂移。我們透過 `evidently` 實作了資料漂移的偵測，並模擬了自動化再訓練的流程。

這只是 MLOps 世界的冰山一角。在真實的生產環境中，你還需要考慮更多細節，例如：
*   如何有效地收集和儲存生產資料？
*   如何設定漂移偵測的閾值？
*   如何處理多個模型和版本？
*   如何進行 A/B 測試來比較新舊模型？
*   如何確保再訓練過程的穩定性和可重複性？

MLOps 的旅程充滿挑戰，但也充滿樂趣！它讓你的機器學習模型真正地「活」起來，並能持續為業務創造價值。繼續探索，你會發現一個更廣闊的 ML 世界！

祝你學習愉快，我們下次見！