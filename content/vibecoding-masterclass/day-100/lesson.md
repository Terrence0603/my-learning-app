好的，我的準學生！🎉

恭喜你！來到程式學習的第 100 天，這真是一個了不起的里程碑！你已經走過了這麼多路，學會了這麼多新知識，現在，我們準備踏入一個更進階、但也超級實用的領域：**MLOps (Machine Learning Operations)**。

別聽到這個名字就緊張，MLOps 聽起來很酷炫，但它的核心理念其實很簡單，就是讓我們的機器學習模型從「實驗室」走向「真實世界」，並且能好好地「照顧」它們，讓它們持續為我們服務。今天，我們就來輕鬆聊聊 MLOps，並用一些簡單的程式碼，讓你感受一下模型生命週期管理與迭代的魅力！

---

## 【第 100 天：實戰：MLOps 模型生命週期管理與迭代策略】

哈囉，未來的 AI 大師！👋

在你學習機器學習的旅程中，可能大部分時間都花在數據準備、模型訓練和評估上了，對吧？這當然非常重要！但你有沒有想過：當你的模型訓練出來後，怎麼讓它持續穩定地提供服務？當數據變了，或者你想嘗試更好的模型，怎麼管理這些「版本」？這就是 MLOps 登場的時候了！

### 為什麼我們需要 MLOps？

想像一下，你辛辛苦苦煮出了一道超級美味的菜（你的 ML 模型），但你不能只煮一次就丟著不管。你得考慮：
1.  **紀錄食譜：** 你用了哪些食材，什麼火候，才能重現這道菜？（**實驗追蹤與重現性**）
2.  **上菜：** 怎麼把菜端給客人，讓他們享用？（**模型部署**）
3.  **客人回饋：** 客人說味道淡了點，下次怎麼改進？（**模型監控與迭代**）
4.  **新食譜：** 如果你發明了新口味，怎麼管理不同版本的食譜？（**模型版本管理**）

MLOps 就是解決這些問題的「餐廳管理系統」！它幫助我們讓 ML 專案從實驗、開發到部署、維護，整個過程都變得更流暢、更可靠。

### MLOps 的核心：模型生命週期與迭代

簡單來說，一個模型的生命週期通常包含幾個階段：
1.  **實驗與開發：** 嘗試不同的數據、特徵、演算法和超參數。
2.  **訓練與評估：** 選擇表現最佳的模型。
3.  **版本管理：** 記錄每個模型版本，包括訓練代碼、數據和性能。
4.  **部署：** 將模型變成可以提供預測服務的 API 或應用。
5.  **監控：** 觀察模型在真實世界中的表現，是否開始「變笨」（數據漂移、模型退化）。
6.  **迭代與再訓練：** 根據監控結果，用新數據或新方法重新訓練和部署新版本。

今天，我們將聚焦在「實驗追蹤」、「模型版本管理」和「迭代」上，並使用一個非常受歡迎的 MLOps 工具：**MLflow**。

### 實戰：用 MLflow 追蹤與迭代模型

MLflow 是一個開源平台，用於管理機器學習的生命週期。它可以幫我們追蹤實驗、打包代碼、管理模型和部署。

**第一步：安裝必要的工具**

請打開你的終端機或命令提示字元，輸入：

```bash
pip install mlflow scikit-learn pandas
```

**第二步：我們的第一個模型實驗與追蹤**

我們來訓練一個簡單的線性迴歸模型，並用 MLflow 記錄下這次實驗的所有資訊。

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 確保 MLflow 不會自動啟動服務，我們稍後手動啟動
# mlflow.set_tracking_uri("http://localhost:5000") # 如果你想連到遠端伺服器

print("--- 實驗開始：訓練第一個模型 ---")

# 1. 準備數據
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 with mlflow.start_run() 來開始一個實驗追蹤會話
with mlflow.start_run(run_name="Initial_Housing_Price_Predictor") as run:
    # 2. 訓練模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 3. 評估模型
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"模型訓練完成！初始 RMSE: {rmse:.2f}")

    # 4. 用 MLflow 記錄超參數和指標
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("noise_level", 10)
    mlflow.log_metric("rmse", rmse)

    # 5. 用 MLflow 記錄模型本身，並註冊到模型註冊表
    # "MyHousingPredictor" 是我們給這個系列模型起的名字
    mlflow.sklearn.log_model(model, "model", registered_model_name="MyHousingPredictor")

    run_id = run.info.run_id
    print(f"MLflow 實驗 ID: {run_id}")
    print("你的第一個模型和實驗記錄已儲存！")

print("\n--- 請在終端機輸入 'mlflow ui' 並訪問 http://localhost:5000 查看結果 ---")
print("你可以在 Models 頁籤看到 'MyHousingPredictor' 的第一個版本。")
```

執行上面的 Python 程式碼後，請**打開一個新的終端機視窗**，然後輸入 `mlflow ui`。接著在瀏覽器中訪問 `http://localhost:5000`。你應該能看到你的實驗記錄，以及在 "Models" 頁籤下多了一個名為 `MyHousingPredictor` 的模型，版本為 1。是不是很酷？你現在已經學會了實驗追蹤和模型版本管理的基本操作！

**第三步：模型迭代策略 - 訓練新版本**

假設我們發現模型表現不夠好，或者我們有了新的數據，決定重新訓練一個更好的模型。MLOps 的精神就是，我們不用覆蓋舊模型，而是訓練一個新版本！

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

print("\n--- 實驗開始：訓練模型的第二個版本 ---")

# 模擬有更多數據，並且數據的噪音更小了 (假設我們做了數據清洗)
X_new, y_new = make_regression(n_samples=150, n_features=1, noise=8, random_state=100) # 更多數據，更少噪音
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=100)

with mlflow.start_run(run_name="Improved_Housing_Price_Predictor") as run:
    # 2. 訓練一個新的模型 (可能是新的演算法，或是用了新數據)
    model_v2 = LinearRegression()
    model_v2.fit(X_train_new, y_train_new)
    predictions_v2 = model_v2.predict(X_test_new)

    # 3. 評估新模型
    rmse_v2 = np.sqrt(mean_squared_error(y_test_new, predictions_v2))
    print(f"新模型訓練完成！改進後 RMSE: {rmse_v2:.2f}")

    # 4. 記錄新模型的超參數和指標
    mlflow.log_param("model_type", "LinearRegression_V2") # 可以是新的超參數組合
    mlflow.log_param("noise_level", 8)
    mlflow.log_metric("rmse", rmse_v2)

    # 5. 將新模型記錄到「同一個」註冊模型名下，它會自動成為新的版本
    mlflow.sklearn.log_model(model_v2, "model_v2", registered_model_name="MyHousingPredictor")

    run_id_v2 = run.info.run_id
    print(f"MLflow 實驗 ID (V2): {run_id_v2}")
    print("你的第二個模型版本已儲存！")

print("\n--- 再次刷新 'mlflow ui' 頁面 ---")
print("現在你應該能在 'MyHousingPredictor' 下看到版本 1 和版本 2 了！")
```

現在，回到你的 `mlflow ui` 介面，刷新頁面。你會發現 `MyHousingPredictor` 模型現在有了兩個版本！你可以比較它們的 RMSE 指標，看看新模型是不是真的改進了。

**第四步：加載並使用特定版本的模型**

在真實世界的應用中，你可能需要根據環境（開發、測試、生產）加載不同版本的模型。MLflow 讓這變得輕而易舉。

```python
import mlflow.pyfunc
import numpy as np

print("\n--- 加載並使用特定版本的模型 ---")

# 加載最新版本的模型 (MLflow 會自動識別哪個是最新)
loaded_model = mlflow.pyfunc.load_model("models:/MyHousingPredictor/latest")
print("已加載 'MyHousingPredictor' 的最新版本。")

# 假設這是我們要預測的新數據點
sample_data = np.array([[12.5]]) # 例如，一個新的房屋特徵

# 使用加載的模型進行預測
prediction = loaded_model.predict(sample_data)
print(f"使用最新模型預測 {sample_data[0][0]} 的房屋價格為: {prediction[0]:.2f}")

# 你也可以加載特定版本，例如版本 1
# loaded_model_v1 = mlflow.pyfunc.load_model("models:/MyHousingPredictor/1")
# prediction_v1 = loaded_model_v1.predict(sample_data)
# print(f"使用版本 1 模型預測 {sample_data[0][0]} 的房屋價格為: {prediction_v1[0]:.2f}")
```

---

### 總結與展望

太棒了！你今天不僅學會了 MLOps 的基本概念，還親手操作了 MLflow 來追蹤實驗、管理模型版本，並體驗了模型迭代的過程。這只是 MLOps 的冰山一角，它還有模型部署、監控、CI/CD 等更多精彩的內容等你探索！

記住，機器學習不是一次性的任務，它是一個持續學習、改進和維護的循環。MLOps 提供的工具和流程，正是幫助我們駕馭這個循環的利器。

第 100 天，你證明了你的堅持和學習能力。繼續保持好奇心，不斷探索，你的程式之路會越走越寬廣！我為你感到驕傲！💪

有任何問題，隨時問我喔！