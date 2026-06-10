好的，程式小夥伴，恭喜你來到我們機器學習旅程的第 38 天！這是一個非常重要的里程碑，因為我們將從單純地「訓練模型」晉升到「專業地管理模型」。今天的課程主題是：**MLOps 實驗追蹤與模型版本管理**。

你可能會想：「MLOps 是什麼？聽起來好高大上！」別擔心，我們會用最輕鬆、最貼近實戰的方式來理解它。想像一下，你不再是單打獨鬥的科學家，而是一個團隊的成員，你需要確保你的實驗結果可重複、可追溯，並且你的模型可以穩定地運行。這就是 MLOps 的核心精神！

---

### 【第 38 天：實戰：MLOps 實驗追蹤與模型版本管理】

嗨，我的程式小英雄！

恭喜你！來到我們機器學習旅程的第 38 天了！走到這裡，你一定已經訓練過不少模型，也感受過從數據中提取知識的樂趣。但你是否曾遇過這些情況：

*   「我上週訓練的那個模型，用了什麼參數來著？」
*   「這個版本的準確率為什麼比上次高？我改了什麼？」
*   「同事說他跑出更好的結果，但我怎麼都重現不出來？」
*   「哪個模型才是現在應該上線的版本啊？」

如果這些問題聽起來很熟悉，那麼，恭喜你！你已經準備好進入 **MLOps (Machine Learning Operations)** 的世界了！MLOps 簡單來說，就是把軟體開發中的最佳實踐（DevOps）應用到機器學習工作流程中，讓我們的模型開發、部署和維護變得更有效率、更可靠。

今天，我們將聚焦於 MLOps 中兩個超級實用的環節：**實驗追蹤 (Experiment Tracking)** 和 **模型版本管理 (Model Versioning)**。而我們的得力助手，就是開源工具 **MLflow**！

---

### 1. 為什麼需要實驗追蹤和模型版本管理？

想像一下，你是一位發明家，正在研發一種新的神奇藥水。如果你每次實驗都不做記錄，只是憑感覺調整配方，那麼當你真的做出有效的藥水時，你怎麼知道它是怎麼來的？又怎麼能確保下次還能做出一樣的藥水呢？

在機器學習的世界裡，模型訓練就是我們的實驗，參數、數據、模型架構就是我們的配方。

*   **實驗追蹤**：就像你的「實驗日誌」。它幫助我們自動記錄每次模型訓練的：
    *   **參數 (Parameters)**：學習率、迭代次數、正則化強度...
    *   **性能指標 (Metrics)**：準確率、精確度、召回率、F1 分數...
    *   **代碼版本 (Code Version)**：通常會連結到 Git commit。
    *   **模型本身 (Artifacts)**：訓練好的模型檔案、圖表、數據預處理器等。
    *   **環境信息 (Environment Information)**：使用的函式庫版本。

    有了這些記錄，我們就可以輕鬆比較不同實驗的結果，找出最佳參數組合，並且隨時回溯任何一個實驗，確保可重複性。

*   **模型版本管理**：當你訓練出一個「合格」的模型後，它就進入了「資產」的範疇。模型版本管理讓我們可以：
    *   為每個模型設定唯一標識，確保每個模型都是可追溯的。
    *   追蹤模型從開發到部署的生命週期（Staging, Production, Archived）。
    *   輕鬆切換、部署不同版本的模型，尤其是在進行 A/B 測試或回滾時非常方便。

總之，它們能讓我們的工作更科學、更高效、更具協作性！

---

### 2. 認識 MLflow：你的 MLOps 小助手

MLflow 是一個開源平台，旨在管理機器學習的整個生命週期。它主要包含四個核心組件：

*   **MLflow Tracking (實驗追蹤)**：記錄和查詢實驗，包括參數、代碼、數據和結果。
*   **MLflow Projects (專案打包)**：將 ML 代碼打包成可重複執行的格式。
*   **MLflow Models (模型格式)**：提供標準的模型打包格式，方便部署到不同平台。
*   **MLflow Model Registry (模型註冊中心)**：集中管理模型的生命週期，追蹤模型版本和階段。

今天，我們主要會用到 **MLflow Tracking** 和 **MLflow Model Registry**。

**安裝 MLflow**

在你的虛擬環境中，執行以下指令：

```bash
pip install mlflow scikit-learn pandas
```

---

### 3. 實戰 MLflow 實驗追蹤

我們將使用一個簡單的邏輯迴歸模型來分類 Iris 數據集，並使用 MLflow 來追蹤我們的實驗。

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

# 啟動 MLflow 自動日誌功能 (針對 scikit-learn)
# mlflow.sklearn.autolog() # 這行可以簡化很多日誌操作，但為了示範，我們手動來

# 1. 準備數據
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 定義實驗參數
C_param = 0.1 # 正則化強度的倒數
solver_param = "liblinear"
max_iter_param = 100

# 3. 使用 mlflow.start_run() 包裹你的訓練代碼
# 這樣 MLflow 就會知道這是一個新的實驗運行
with mlflow.start_run() as run:
    # 記錄參數
    mlflow.log_param("C", C_param)
    mlflow.log_param("solver", solver_param)
    mlflow.log_param("max_iter", max_iter_param)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # 訓練模型
    model = LogisticRegression(C=C_param, solver=solver_param, max_iter=max_iter_param)
    model.fit(X_train, y_train)

    # 進行預測
    y_pred = model.predict(X_test)

    # 計算性能指標
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # 記錄指標
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 記錄模型
    # "logistic_regression_model" 是模型在 MLflow 追蹤伺服器中的 artifact path
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    # 獲取當前運行的 run_id，之後模型註冊會用到
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id}")
    print(f"Accuracy: {accuracy}")
    print(f"Model logged to: runs:/{run_id}/logistic_regression_model")

print("實驗追蹤完成！請在終端機中輸入 'mlflow ui' 查看結果。")
```

**運行你的 MLflow UI**

1.  將上面的程式碼儲存為 `mlflow_tracking_example.py`。
2.  在終端機中，導航到該檔案所在的目錄。
3.  執行 `python mlflow_tracking_example.py`。
4.  等待程式執行完成後，在**同一個終端機**或**新開一個終端機**，輸入：
    ```bash
    mlflow ui
    ```
5.  你的瀏覽器會自動打開（或你手動訪問 `http://localhost:5000`），你就能看到 MLflow 的使用者介面了！在這裡，你可以看到每次實驗的列表，點擊進去可以查看詳細的參數、指標和模型檔案。是不是很酷？

多執行幾次你的 Python 程式，每次稍微修改 `C_param` 或 `solver_param`，然後重新整理 `mlflow ui` 頁面，你就能看到新的實驗記錄！

---

### 4. 模型版本管理初探 (MLflow Model Registry)

當你透過 MLflow Tracking 找到一個性能優異、滿意的模型後，你可能會想把它「正式註冊」起來，方便追蹤它的生命週期和版本。這就是 MLflow Model Registry 的功能。

我們可以在上面的程式碼的基礎上，新增註冊模型的步驟。在 `with mlflow.start_run()` 區塊**之外**，但程式執行完畢之後：

```python
# ... (上面的訓練和追蹤代碼) ...

# 現在我們來註冊這個模型到 MLflow Model Registry
# 注意：你需要先啟動 MLflow Tracking Server 並設定後端儲存，
# 這裡我們用預設的 file-based，需要確保 artifact_uri 可訪問。

# 定義模型名稱
registered_model_name = "IrisLogisticRegressionModel"

# 構建模型 URI，它指向我們剛才日誌的模型
# run_id 是上一步驟中獲取的
model_uri = f"runs:/{run_id}/logistic_regression_model"

print(f"\n正在註冊模型 '{registered_model_name}' (來源: {model_uri})...")

# 註冊模型
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(f"模型已註冊！版本: {registered_model.version}")
print(f"你可以前往 MLflow UI 的 'Models' 頁面查看。")

# 你也可以透過以下方式加載已註冊的模型 (例如加載最新版本)
# loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{registered_model_name}/latest")
# print(f"已成功加載註冊模型 (版本 {registered_model.version})！")
```

重新執行你的 Python 程式，然後再打開 `mlflow ui`。這次，除了在 `Experiments` 頁面看到你的實驗運行，你還會看到一個 `Models` 的頁籤。點擊它，你就能看到你剛剛註冊的 `IrisLogisticRegressionModel`，以及它的第一個版本！每次你再次運行程式並註冊，就會生成新的版本。

---

### 結語

恭喜你！今天我們一起打開了 MLOps 的大門，學習了如何使用 MLflow 進行實驗追蹤和模型版本管理。這不僅讓你的機器學習工作變得更有組織、更專業，也為未來與團隊協作、模型部署打下了堅實的基礎。

MLOps 是一個廣闊的領域，今天我們只是踏出了第一步。未來你還可以探索更多 MLflow 的功能，或是其他 MLOps 工具，讓你的 AI 模型從開發到生產的整個旅程更加順暢。

別害怕嘗試，多動手實踐，你很快就能成為 MLOps 的高手！我們第 39 天見！