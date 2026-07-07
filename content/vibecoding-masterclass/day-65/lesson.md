太棒了！能夠堅持到第 65 天，代表你對程式的熱情和毅力都非常驚人！今天我們要探索一個超級實用，能讓你的機器學習模型從實驗室走向真實世界的關鍵環節：MLOps 的模型部署策略與版本管理。別擔心，這聽起來很專業，但我們會用最輕鬆的方式來揭開它的神秘面紗！

---

## 【第 65 天：實戰：MLOps 模型部署策略與版本管理】

嗨，未來的 MLOps 大師！

恭喜你！走到第 65 天，你已經是個貨真價實的程式開發者了！我們之前學會了如何訓練模型，讓它們學會預測、分類。但你知道嗎？訓練好模型只是故事的一半！真正厲害的是，如何把你辛辛苦苦訓練出來的模型，安全、穩定、有效率地「交到」使用者手上，讓它在實際應用中發揮價值。這就是 MLOps (Machine Learning Operations) 的魅力所在，特別是今天的重點：**模型部署策略與版本管理**！

想像一下，你的模型就像是精心製作的產品，而 MLOps 就是確保這個產品能順利從工廠（訓練環境）運送到顧客手中（生產環境），並且能隨時更新、替換的物流系統。

### 為什麼模型部署和版本管理這麼重要？

1.  **不停進化：** 數據會變，需求會變，模型也需要不斷更新和改進。如果沒有好的版本管理，你怎麼知道哪個版本最好？
2.  **避免錯誤：** 部署新模型時，你總不希望一上線就出問題，影響到所有用戶吧？部署策略就是幫你「安全換胎」的技巧。
3.  **可追溯性：** 當模型表現不如預期時，你能迅速知道是哪個版本的模型、哪個版本的程式碼、甚至哪個版本的數據導致的問題嗎？版本管理幫你記下這一切。
4.  **團隊協作：** 在團隊中，每個人都清楚正在使用哪個模型版本，避免混亂。

今天，我們會用一個簡單的例子，帶你一窺 MLOps 的部署和版本管理的精髓，並實際使用 **MLflow** 這個超棒的工具來管理我們的模型。

### 實戰範例：用 MLflow 進行模型版本管理

MLflow 是一個開源平台，用於管理機器學習的生命週期，其中包含了一個強大的 **Model Registry** (模型註冊中心)，可以讓我們輕鬆地對模型進行版本控制。

#### 步驟一：訓練模型並註冊到 MLflow

首先，我們需要一個簡單的模型。我們將訓練一個 Logistic Regression 模型來做分類，然後將它存入 MLflow Model Registry。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification # 用來生成假數據

# 設定 MLflow Tracking URI，如果沒有設定，MLflow 會在本地當前目錄建立一個 'mlruns' 資料夾
# 這裡我們使用本地文件系統作為追蹤伺服器
mlflow.set_tracking_uri("file:///./mlruns")

# 我們給模型一個好記的名字
model_name = "SimpleClassificationModel"

# 1. 準備數據 (使用 make_classification 生成一些假數據)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 訓練模型
print("--- 訓練第一個模型版本 (v1) ---")
with mlflow.start_run(run_name="Training_V1"):
    model_v1 = LogisticRegression(solver='liblinear', random_state=42)
    model_v1.fit(X_train, y_train)
    y_pred_v1 = model_v1.predict(X_test)
    accuracy_v1 = accuracy_score(y_test, y_pred_v1)

    print(f"模型 V1 準確度: {accuracy_v1:.4f}")

    # 紀錄參數和指標
    mlflow.log_param("solver", "liblinear")
    mlflow.log_metric("accuracy", accuracy_v1)

    # 3. 將模型註冊到 MLflow Model Registry
    # artifact_path 是模型在 run 裡的子路徑
    # registered_model_name 是在 Model Registry 裡顯示的名稱
    mlflow.sklearn.log_model(
        sk_model=model_v1,
        artifact_path="model",
        registered_model_name=model_name
    )
    print(f"模型 '{model_name}' 已註冊為 V1。")

print("\n--- 訓練第二個模型版本 (v2) ---")
# 假設我們改進了模型參數，或者用了不同的算法
with mlflow.start_run(run_name="Training_V2"):
    model_v2 = LogisticRegression(solver='saga', max_iter=200, random_state=42) # 稍微調整參數
    model_v2.fit(X_train, y_train)
    y_pred_v2 = model_v2.predict(X_test)
    accuracy_v2 = accuracy_score(y_test, y_pred_v2)

    print(f"模型 V2 準確度: {accuracy_v2:.4f}")

    mlflow.log_param("solver", "saga")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", accuracy_v2)

    mlflow.sklearn.log_model(
        sk_model=model_v2,
        artifact_path="model",
        registered_model_name=model_name # 再次使用相同的名稱，MLflow 會自動建立新版本
    )
    print(f"模型 '{model_name}' 已註冊為 V2。")

# 你可以透過終端機執行 'mlflow ui' 指令，然後打開 http://localhost:5000 查看結果！
# 在 Models 頁籤下，你會看到 'SimpleClassificationModel' 以及它的不同版本。
```

執行這段程式碼後，MLflow 會在本地創建一個 `mlruns` 資料夾，並將模型的訓練過程和模型檔案都儲存起來。最重要的是，你的模型將以 `SimpleClassificationModel` 的名稱，依序有了 V1 和 V2 兩個版本。

#### 步驟二：部署策略：選擇與加載特定版本的模型

在實際部署時，我們不會直接拿著訓練好的模型檔案，而是從 Model Registry 中加載特定版本的模型。這給了我們極大的彈性！

```python
import mlflow.pyfunc

# 我們要從 Model Registry 加載的模型名稱
model_name = "SimpleClassificationModel"

# 1. 加載最新版本的模型 (例如用於開發測試)
print("\n--- 加載最新版本的模型 ---")
latest_model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
print(f"成功加載模型: {model_name} 的最新版本。")

# 你可以使用 latest_model 進行預測
# latest_model.predict(...)

# 2. 加載特定版本的模型 (例如部署到生產環境 V1，測試環境 V2)
print("\n--- 加載特定版本的模型 (v1) ---")
model_version_1 = mlflow.pyfunc.load_model(f"models:/{model_name}/1")
print(f"成功加載模型: {model_name} 的 V1 版本。")

print("\n--- 加載特定版本的模型 (v2) ---")
model_version_2 = mlflow.pyfunc.load_model(f"models:/{model_name}/2")
print(f"成功加載模型: {model_name} 的 V2 版本。")

# 假設現在 V1 是生產環境的模型
# 而 V2 正在測試中，準備替換 V1
def predict_with_production_model(data):
    # 在真實世界中，這裡會透過 API 服務來處理請求
    return model_version_1.predict(data)

# 我們來測試一下，用 V1 模型對一些新數據進行預測
# 注意：這裡的 `X_test` 應該是新的、從未見過的數據，為簡化我們仍使用它
sample_data = X_test[:5]
predictions_v1 = predict_with_production_model(sample_data)
print(f"\n使用 V1 模型對樣例數據的預測: {predictions_v1}")

# 假設現在 V2 模型測試完成，表現更好，我們要把它部署為新的生產模型
# 在 MLflow UI 中，你可以將 V2 標記為 'Production' Stage
# 然後你的部署服務就可以加載 f"models:/{model_name}/Production"
# 這是一個非常棒的部署策略，稱為「藍綠部署」或「金絲雀部署」的基礎
print("\n--- 想像你將 V2 標記為生產環境模型 ---")
# 從 MLflow Registry 中，將 V2 模型的 Stage 設置為 Production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Production"
)
print(f"模型 {model_name} V2 版本已轉為 'Production' Stage。")

# 現在你可以加載 'Production' 階段的模型
model_in_production = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
predictions_prod_new = model_in_production.predict(sample_data)
print(f"使用新的生產模型 (V2) 對樣例數據的預測: {predictions_prod_new}")

```

#### 部署策略小補充：

*   **藍綠部署 (Blue-Green Deployment):**
    想像你有兩套一模一樣的環境，一套是「藍」環境 (目前線上的 V1 模型)，一套是「綠」環境 (部署新模型 V2)。當你把 V2 部署到綠環境測試沒問題後，直接把流量從藍環境切換到綠環境。如果新模型有問題，可以快速切回藍環境，用戶幾乎無感。
*   **金絲雀部署 (Canary Deployment):**
    這個名字來自以前礦工會帶金絲雀進礦坑探測毒氣。金絲雀部署就是先將新模型部署到一小部分用戶，觀察模型表現。如果沒問題，再逐步擴大用戶群。這樣即使新模型有問題，也只影響少數用戶。

MLflow 的 Model Registry 配合 `models:/<model-name>/<stage>` (例如 `models:/SimpleClassificationModel/Production`) 這樣的加載方式，就為你實現這些部署策略奠定了堅實的基礎！你可以在 MLflow UI 中手動將模型版本從 `None` 階段轉換到 `Staging` (測試)、`Production` (生產) 或 `Archived` (歸檔)。

### 總結與鼓勵

哇，今天我們學習了 MLOps 中非常重要的一環：模型部署策略與版本管理！我們用 MLflow 實際操作了模型的註冊、不同版本的管理，以及如何加載它們。這讓你能夠：

*   **系統化地管理你的模型資產。**
*   **安全地更新你的線上模型。**
*   **為未來的自動化部署打下基礎。**

MLOps 是一個廣闊的領域，但你已經跨出了關鍵的第一步。從現在開始，當你訓練完一個模型，不僅要想著它有多「準」，更要想想如何讓它安全、穩定地「服務」千千萬萬的使用者！

別怕複雜，我們總是一步一步來。每一次的學習，都在為你的 MLOps 超能力充能！繼續加油，你一定能成為一位出色的機器學習工程師！