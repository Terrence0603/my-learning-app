哈囉，我的程式學習夥伴們！歡迎來到【第 81 天】的旅程！

恭喜你們走到這一步，想必對模型訓練已經駕輕就熟了。你可能已經訓練出好幾個厲害的模型，它們在測試集上表現亮眼。但現在問題來了：當你訓練了十個、二十個，甚至上百個模型時，你還記得哪個模型是用哪組參數訓練的？哪個模型的表現最好？哪個版本是目前線上部署的？

是不是覺得有點混亂了？沒錯！這就是我們今天要解決的痛點：**模型版本管理與模型註冊中心**！今天我們要學會如何像專業人士一樣，管理你的模型資產，告別混亂，成為模型管理大師！

---

## 主題：【第 81 天：實戰：模型版本管理與模型註冊中心】

### 🎯 學習目標

*   理解為什麼需要模型版本管理。
*   了解模型註冊中心的核心概念與好處。
*   透過 MLflow 實作模型追蹤、版本註冊與加載。

### 🚀 告別混亂，擁抱秩序：為什麼需要模型版本管理？

想像一下，你是一位非常厲害的廚師，每天都會創造新的食譜。如果你不把每個食譜的版本、配料、製作方法都記錄下來，一旦有人問你「上次那個超好吃的蛋糕怎麼做的？」，你可能就傻眼了！

在機器學習的世界裡，模型就像我們的食譜：

1.  **可重現性 (Reproducibility)**：你用的數據集是什麼？調整了哪些超參數？模型表現如何？如果沒有記錄，下次很難復現同一個模型。
2.  **協作開發 (Collaboration)**：團隊成員之間如何共享和使用不同的模型版本？總不能靠口耳相傳或命名 `model_final.pkl`, `model_final_really.pkl` 吧？
3.  **部署與回滾 (Deployment & Rollback)**：哪個版本是目前線上使用的？如果新版本出問題，如何快速回溯到舊的穩定版本？
4.  **審計與監管 (Auditing & Governance)**：在某些產業，你需要追蹤模型的每個變動，以符合法規要求。

這時候，**模型版本管理**就變得至關重要了！它能確保你的每個模型都有唯一的身份證，記錄了它的出生證明、成長歷程和表現。

### 🏠 模型的中央圖書館：模型註冊中心 (Model Registry)

模型註冊中心就像是一個專為模型設計的中央圖書館或博物館。它是一個集中的儲存庫，讓你可以：

*   **註冊 (Register)** 模型，賦予它一個唯一的名稱。
*   **追蹤版本 (Version)**：每次訓練出一個新模型，就給它一個新的版本號 (v1, v2, v3...)。
*   **管理元數據 (Manage Metadata)**：記錄模型的訓練參數、指標、訓練日期、作者等資訊。
*   **定義階段 (Define Stages)**：標註模型所處的生命週期階段，例如：
    *   `Staging` (測試階段)：正在測試中的模型。
    *   `Production` (生產階段)：目前線上部署的模型。
    *   `Archived` (歸檔)：不再使用的舊模型。
*   **輕鬆加載 (Load Easily)**：無論何時何地，都能透過名稱和版本號輕鬆加載任何一個模型。

市面上有許多工具可以做到這點，其中一個非常流行且好用的就是 **MLflow**！它提供了一個完整的 MLOps 生命週期管理方案，包含了追蹤 (Tracking)、專案 (Projects)、模型 (Models) 和註冊中心 (Model Registry)。

### 🛠️ 動手實作：使用 MLflow 進行模型版本管理

我們將使用 MLflow 來實踐模型註冊中心的功能。

首先，確保你已經安裝了必要的套件：

```bash
pip install mlflow scikit-learn pandas
```

**步驟一：訓練模型並註冊**

我們來訓練一個簡單的邏輯迴歸模型，並將它註冊到 MLflow Registry 中。

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np

# 建立一些假資料
np.random.seed(42)
data = {
    'feature_1': np.random.rand(100),
    'feature_2': np.random.rand(100) * 10,
    'target': np.random.randint(0, 2, 100) # 0或1
}
df = pd.DataFrame(data)

X = df[['feature_1', 'feature_2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義模型名稱
registered_model_name = "SimpleLogisticModel"

# 啟動 MLflow 追蹤
with mlflow.start_run(run_name="LogisticRegression_V1") as run:
    # 定義模型參數
    solver = "liblinear"
    max_iter = 100

    # 訓練模型
    model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)

    # 進行預測
    y_pred = model.predict(X_test)

    # 計算指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    # 記錄參數、指標
    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 記錄模型並註冊到 Model Registry
    # 當 registered_model_name 參數被設定時，MLflow 會自動將模型註冊到 Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_model",
        registered_model_name=registered_model_name,
        # tags={"model_type": "classification", "data_source": "synthetic"} # 可以加上標籤
    )

    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"模型 '{registered_model_name}' 已註冊為版本 1。")

print("\n--- 嘗試第二次訓練，並註冊為新版本 ---")

# 再次訓練，修改參數，模擬新版本的產生
with mlflow.start_run(run_name="LogisticRegression_V2") as run:
    solver = "lbfgs" # 改變 solver 參數
    max_iter = 200 # 改變 max_iter 參數

    model_v2 = LogisticRegression(solver=solver, max_iter=max_iter, random_state=42)
    model_v2.fit(X_train, y_train)
    y_pred_v2 = model_v2.predict(X_test)

    accuracy_v2 = accuracy_score(y_test, y_pred_v2)
    precision_v2 = precision_score(y_test, y_pred_v2)
    recall_v2 = recall_score(y_test, y_pred_v2)

    print(f"Metrics V2 - Accuracy: {accuracy_v2}, Precision: {precision_v2}, Recall: {recall_v2}")

    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", accuracy_v2)
    mlflow.log_metric("precision", precision_v2)
    mlflow.log_metric("recall", recall_v2)

    mlflow.sklearn.log_model(
        sk_model=model_v2,
        artifact_path="logistic_model_v2",
        registered_model_name=registered_model_name, # 繼續使用相同名稱，MLflow 會自動建立新版本
    )
    print(f"模型 '{registered_model_name}' 已註冊為新版本 (版本 2)。")

```

執行上面的程式碼後，MLflow 會啟動一個本地的追蹤伺服器（如果沒有指定遠端伺服器）。你可以在終端機執行 `mlflow ui` 命令，然後在瀏覽器打開 `http://localhost:5000` 來查看你的 MLflow UI。

在 MLflow UI 中，點擊左側導航欄的 `Models` 選項。你會看到你的 `SimpleLogisticModel`，並且它會有兩個版本 (Version 1 和 Version 2)。你可以點進去查看每個版本的詳細資訊，包括它的訓練參數、指標，以及它目前所處的階段 (`None` 代表未設定)。

**步驟二：管理模型階段 (Stages)**

在 MLflow UI 的 `Models` 頁面，你可以手動將模型的階段從 `None` 轉換為 `Staging` 或 `Production`。例如，你可以點擊 `SimpleLogisticModel` 的 Version 1，然後在頁面右上方選擇 `Stage`，將它改為 `Production`。同樣地，你可以將 Version 2 改為 `Staging`。

這是一個非常直觀且重要的功能，讓你和你的團隊清楚知道哪個模型是目前生產環境使用的。

**步驟三：加載已註冊的模型**

現在，我們可以在程式碼中輕鬆加載任何一個已註冊的模型，無論它是特定版本，還是特定階段的模型。

```python
import mlflow.pyfunc # 用於加載 MLflow 儲存的模型

# 加載目前處於 "Production" 階段的模型
production_model_uri = f"models:/{registered_model_name}/Production"
loaded_production_model = mlflow.pyfunc.load_model(production_model_uri)
print(f"\n已加載 'Production' 階段的模型。")

# 使用加載的模型進行預測
sample_data = pd.DataFrame([[0.5, 5.0]], columns=['feature_1', 'feature_2'])
prediction_prod = loaded_production_model.predict(sample_data)
print(f"使用 Production 模型預測: {prediction_prod}")

# 加載特定版本的模型 (例如，版本 2)
version_2_model_uri = f"models:/{registered_model_name}/2" # 或者 "Latest"
loaded_version_2_model = mlflow.pyfunc.load_model(version_2_model_uri)
print(f"\n已加載版本 2 的模型。")
prediction_v2 = loaded_version_2_model.predict(sample_data)
print(f"使用版本 2 模型預測: {prediction_v2}")

# 加載最新的模型
latest_model_uri = f"models:/{registered_model_name}/Latest"
loaded_latest_model = mlflow.pyfunc.load_model(latest_model_uri)
print(f"\n已加載最新版本的模型。")
prediction_latest = loaded_latest_model.predict(sample_data)
print(f"使用最新模型預測: {prediction_latest}")
```

是不是超級方便！你不需要記住檔案路徑，只需要模型的名稱和版本/階段，MLflow 就能幫你找到並加載它。這對模型部署、A/B 測試和團隊協作來說是巨大的幫助！

---

### 💡 總結與鼓勵

恭喜你！今天你學會了機器學習工程中一個非常重要的概念：**模型版本管理與模型註冊中心**。透過 MLflow，我們不僅能追蹤模型的訓練過程，還能將模型註冊、版本化，並根據其生命週期定義階段，最終輕鬆地加載和使用它們。

這一步讓你從一個單純的模型訓練者，轉變成一個懂得如何管理和部署模型，更具專業素養的 MLOps 實踐者！這是一項非常有價值的技能，會讓你的機器學習專案更加有條不紊、更具效率。

繼續探索 MLflow 的更多功能，嘗試將你的真實專案模型註冊起來。你會發現它能為你省下大量的時間和精力！

今天的學習就到這裡，我們【第 82 天】再見！繼續加油，我的學習夥伴！💪