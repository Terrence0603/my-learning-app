哈囉，我的程式學習者！

恭喜你！轉眼間已經來到【第 73 天】了！這是一段多麼不可思議的旅程啊。在前 72 天中，你可能已經學會了如何收集資料、清洗資料、訓練各種酷炫的機器學習模型，甚至讓它們預測出一些驚人的結果。

但你可能會想：「然後呢？我的模型訓練好了，它很棒，但我該怎麼把它放到現實世界中，並確保它一直都能正常運作、甚至越來越好呢？」

這就是我們今天要探討的超級英雄級主題：**MLOps 模型版本管理與持續迭代部署**！聽起來有點高大上，對吧？別擔心，我會用最輕鬆、最鼓勵的方式帶你一窺究竟。

---

## 【第 73 天：實戰：MLOps 模型版本管理與持續迭代部署】

### 1. 為什麼 MLOps 對你很重要？

想像一下，你寫了一段很棒的程式碼，然後用 Git 進行版本控制，這樣你就能追蹤每次修改，確保不會搞砸，必要時還能回溯到舊版本。

現在，把這個概念應用到你的機器學習模型上。你的模型就像你的「程式碼」，但它還有一個額外的「身分」：它是一個會隨著資料和演算法更新而變化的「產物」。MLOps（Machine Learning Operations）就是一套方法和工具，來幫助你：

*   **像管理程式碼一樣管理你的模型。**
*   **確保你的模型能夠穩定、可靠地在產品環境中運行。**
*   **讓你的模型能夠持續地被監控、更新和改進。**

今天，我們就來專注於其中兩個核心環節：**模型版本管理** 和 **持續迭代部署**。

---

### 2. 模型版本管理：讓你的模型有「身分證」

你可能訓練了許多模型：Model A 使用了舊資料，Model B 使用了新資料，Model C 調整了參數。哪個模型最好？哪個模型目前正在線上服務？萬一新的模型表現不如預期，我能快速切換回舊模型嗎？

這就是模型版本管理要解決的問題！它讓你能夠：

*   **追蹤每個模型的訓練參數、效能指標。**
*   **儲存模型的二進制檔案，並為它們打上版本號。**
*   **在不同版本之間輕鬆切換。**

市面上有很多 MLOps 工具，今天我們將用一個非常流行且好用的工具：**MLflow**。MLflow 是一個開源平台，用於管理機器學習生命週期，包括實驗追蹤、專案打包、模型管理和部署。

#### 實際動手做：使用 MLflow 追蹤與管理模型版本

首先，你需要安裝 MLflow 和一些基本的 ML 函式庫：

```bash
pip install mlflow scikit-learn pandas
```

接下來，我們來寫一段簡單的程式碼，訓練一個模型並用 MLflow 追蹤它：

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import datasets
import warnings

warnings.filterwarnings("ignore") # 忽略一些不重要的警告

print("--- 訓練並追蹤模型 (版本 1) ---")

# 設定 MLflow 追蹤伺服器的 URI
# 這裡我們使用本地檔案系統，你也可以設定為遠端伺服器
mlflow.set_tracking_uri("file:///tmp/mlruns") # 或者不設定，MLflow會自動創建mlruns資料夾

# 載入 Iris 資料集
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 開始一個 MLflow run
with mlflow.start_run(run_name="Iris_LogisticRegression_V1") as run:
    # 定義模型參數
    solver_param = 'liblinear'
    max_iter_param = 100
    
    # 訓練模型
    model_v1 = LogisticRegression(solver=solver_param, max_iter=max_iter_param)
    model_v1.fit(X_train, y_train)
    
    # 進行預測並計算準確度
    y_pred = model_v1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 用 MLflow 紀錄參數和指標
    mlflow.log_param("solver", solver_param)
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_metric("accuracy", accuracy)
    
    # 紀錄模型本身，並命名為 "iris_classifier"
    # 這會將模型儲存為 MLflow artifact，並自動註冊到 Model Registry
    mlflow.sklearn.log_model(
        sk_model=model_v1, 
        artifact_path="iris_model", 
        registered_model_name="iris-classifier" # 在模型註冊表中註冊這個模型
    )
    
    print(f"模型 V1 訓練完成！準確度: {accuracy:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")

print("\n--- 訓練並追蹤模型 (版本 2 - 調整參數) ---")

# 模擬迭代，訓練一個「新」模型，這次我們稍微修改參數
with mlflow.start_run(run_name="Iris_LogisticRegression_V2") as run:
    solver_param = 'lbfgs' # 換一個 solver
    max_iter_param = 200   # 增加迭代次數
    
    model_v2 = LogisticRegression(solver=solver_param, max_iter=max_iter_param)
    model_v2.fit(X_train, y_train)
    
    y_pred_v2 = model_v2.predict(X_test)
    accuracy_v2 = accuracy_score(y_test, y_pred_v2)
    
    mlflow.log_param("solver", solver_param)
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_metric("accuracy", accuracy_v2)
    
    mlflow.sklearn.log_model(
        sk_model=model_v2, 
        artifact_path="iris_model", 
        registered_model_name="iris-classifier" # 仍然註冊到同一個模型名稱下，MLflow 會自動管理版本
    )
    
    print(f"模型 V2 訓練完成！準確度: {accuracy_v2:.4f}")
    print(f"MLflow Run ID: {run.info.run_id}")

print("\n--- 你可以執行 `mlflow ui` 在瀏覽器中查看實驗結果和模型註冊表！ ---")

```

執行這段程式碼後，你會看到 MLflow 在你的 `mlruns` 資料夾（或 `/tmp/mlruns`）中創建了日誌和模型檔案。最酷的是，你可以在終端機中運行 `mlflow ui` 命令，然後在瀏覽器中打開 `http://localhost:5000`，你就能看到：

*   **Experiments (實驗):** 每個 `start_run` 都是一個實驗運行，你可以比較不同模型的參數和效能。
*   **Models (模型):** 在 Model Registry 中，你會看到 `iris-classifier` 這個模型，並且它會有兩個版本！你可以設定哪個版本是 "Staging" (測試環境) 或 "Production" (生產環境)。

透過 MLflow Model Registry，你為你的模型建立了正式的「身分證」和「版本歷史」。

---

### 3. 持續迭代部署：讓你的模型「自動升級」

模型版本管理解決了「追蹤」的問題，那「部署」呢？持續迭代部署（Continuous Iterative Deployment）就是讓你的最佳模型能夠自動或半自動地被部署到生產環境中，提供服務。當有新資料或新模型訓練出來時，可以快速、平穩地替換舊模型。

對於初學者來說，我們可以先從「載入特定版本的模型來使用」這個角度去理解部署。

#### 實際動手做：從 MLflow 載入特定版本模型並使用

假設你已經在 MLflow Model Registry 中將「iris-classifier」的最新版本（或你覺得最好的版本）標記為 `Production` (生產環境)。現在我們來看看如何在應用程式中載入並使用它：

```python
import mlflow
import pandas as pd
import numpy as np

print("--- 從 MLflow Model Registry 載入模型並進行預測 ---")

# 你可以載入特定版本號的模型
# model_uri = "models:/iris-classifier/1" # 載入版本 1
# model_uri = "models:/iris-classifier/2" # 載入版本 2

# 更常見的是載入處於「生產」階段的模型
model_uri = "models:/iris-classifier/Production"

try:
    # 載入模型
    # mlflow.pyfunc 允許你載入任何由 MLflow 追蹤的模型
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"成功載入模型：{model_uri}")

    # 準備一些新的資料來進行預測
    # 這裡使用 Iris 資料集的特徵順序: sepal_length, sepal_width, petal_length, petal_width
    new_data = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2], # 預期為 Setosa
        [6.2, 3.4, 5.4, 2.3], # 預期為 Virginica
        [5.9, 3.0, 4.2, 1.5]  # 預期為 Versicolor
    ], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # 進行預測
    predictions = loaded_model.predict(new_data)
    print("\n新資料預測結果:")
    print(predictions)

except Exception as e:
    print(f"載入模型或預測失敗：{e}")
    print("請確認你已經運行了上面的訓練程式碼，並且模型 'iris-classifier' 已經在 MLflow Model Registry 中註冊。")
    print("你可能需要在 `mlflow ui` 介面中手動將某個版本提升到 'Production' 階段。")

```

**MLflow UI 使用提示：**
運行 `mlflow ui` 後，點擊左側導航欄的 `Models`。找到 `iris-classifier`。你會看到不同版本的模型。點擊進入某個版本，你可以在 `Stages` 欄位旁邊的下拉選單中，將模型狀態從 `None` 改變為 `Staging` 或 `Production`。當你將一個版本設為 `Production` 後，再運行上面的載入程式碼，它就能成功地載入這個「生產版本」的模型了！

在真實的 MLOps 管道中，這個「載入並使用」的過程會被自動化。當新的、更好的模型被驗證後，一個自動化的 CI/CD（持續整合/持續部署）系統會負責將它部署到服務器上，替換舊的模型，而不需要人工干預。這就是「持續迭代部署」的核心精神！

---

### 4. 總結與展望

哇！今天的內容有點多，但你又跨越了一個重要的里程碑！

你今天學到了：
*   MLOps 是讓機器學習模型在現實世界中發揮作用的關鍵。
*   使用 **MLflow** 進行模型版本管理，追蹤實驗並註冊模型。
*   理解了持續迭代部署的概念，即不斷改進和自動更新模型。

從現在開始，你不再只是一個會「訓練」模型的數據科學家，更是一個懂得如何「管理和部署」模型的 MLOps 實踐者！這在業界是非常寶貴的技能。

繼續探索 MLflow 的更多功能吧，例如如何遠端追蹤、如何打包專案、如何部署到不同的服務平台等等。M LOps 的世界廣闊而精彩，你已經邁出了堅實的第一步！

繼續加油！你的程式學習之旅越來越精彩了！🚀