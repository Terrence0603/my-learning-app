太棒了！第 128 天，你已經在機器學習的旅程中累積了扎實的基礎。現在，我們不只會訓練模型，還要學習如何像專業人士一樣，管理這些珍貴的模型資產。今天的主題是 MLOps 的核心環節之一：**模型版本控制與管理**！

---

## 第 128 天：實戰：MLOps 模型版本控制與管理

嗨，親愛的學習者！

恭喜你走到第 128 天！回想一下，我們從基礎的資料處理、模型訓練，一路走來，你已經成功地讓電腦從資料中學習。但你有沒有想過，當你訓練了十幾個、幾十個甚至幾百個模型時，該如何追蹤哪一個模型的表現最好？哪一個模型使用了哪些參數？更重要的是，當一個模型在實際應用中表現不佳時，如何快速回溯到一個穩定可靠的舊版本？

這就是我們今天的主題——**MLOps (Machine Learning Operations) 模型版本控制與管理**所要解決的問題！別擔心，聽起來可能有點複雜，但我們會用輕鬆愉快的方式，搭配一個非常實用的工具來實戰。

### 為什麼需要模型版本控制？

想像一下，你寫程式會用 Git 來管理程式碼的版本，對吧？這樣無論你做了多少修改，都能追溯、比較、甚至回到某個舊版本。對於機器學習模型來說，這個需求更是迫切！

1.  **重現性 (Reproducibility)**：如果沒有版本控制，你很難知道一個模型是如何被訓練出來的（用了什麼資料、什麼參數、什麼程式碼版本）。
2.  **效能追蹤 (Performance Tracking)**：隨著時間推移，模型的效能可能會因為資料變化而下降 (Data Drift)。你需要追蹤不同模型的表現，並知道哪個版本是目前最優的。
3.  **協作 (Collaboration)**：團隊合作時，每個人都可能訓練出新模型。如何整合、比較大家的成果？
4.  **回溯與部署 (Rollback & Deployment)**：當新模型出現問題時，能夠迅速回溯到舊版本。選擇最佳模型部署到生產環境。

這一切，都是 MLOps 的魅力所在。而今天，我們要介紹一個開源且功能強大的工具：**MLflow**。

### 認識 MLflow：你的模型管理好幫手

MLflow 是一個專為機器學習生命週期設計的平台，它提供了四個主要組件：

*   **Tracking (追蹤)**：記錄實驗、參數、指標和模型。
*   **Projects (專案)**：將程式碼打包成可重現的格式。
*   **Models (模型)**：標準化打包模型，使其可以在各種環境中部署。
*   **Model Registry (模型註冊中心)**：集中管理模型的生命週期，包括版本控制、階段轉換（例如從「Staging」到「Production」）。

今天，我們主要會聚焦在 **Tracking** 和 **Model Registry**。

### 前置準備

在開始之前，請確保你已經安裝了必要的函式庫：

```bash
pip install mlflow scikit-learn pandas
```

### 實戰：追蹤與註冊你的模型

現在，我們來訓練一個簡單的分類模型，並利用 MLflow 來追蹤它的資訊並將其註冊到模型註冊中心。

我們會使用 Iris (鳶尾花) 資料集，這是一個經典的多類別分類問題。

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# 1. 啟動 MLflow UI (如果你還沒啟動的話，可以在終端機執行 `mlflow ui`)
#    MLflow 會自動在本地的 ./mlruns 資料夾中建立一個追蹤資料庫。

# 2. 載入資料
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 第一次訓練模型 (版本 1) ---
print("--- 訓練模型版本 1 ---")
with mlflow.start_run(run_name="Iris_Classifier_V1") as run:
    # 定義模型參數
    n_estimators_v1 = 100
    max_depth_v1 = 10
    random_state_v1 = 42

    # 訓練模型
    model_v1 = RandomForestClassifier(
        n_estimators=n_estimators_v1,
        max_depth=max_depth_v1,
        random_state=random_state_v1
    )
    model_v1.fit(X_train, y_train)

    # 預測與評估
    y_pred_v1 = model_v1.predict(X_test)
    accuracy_v1 = accuracy_score(y_test, y_pred_v1)
    
    # Log 參數到 MLflow
    mlflow.log_param("n_estimators", n_estimators_v1)
    mlflow.log_param("max_depth", max_depth_v1)
    mlflow.log_param("random_state", random_state_v1)
    
    # Log 指標到 MLflow
    mlflow.log_metric("accuracy", accuracy_v1)
    
    # 將模型儲存並註冊到 MLflow Model Registry
    # 'IrisRandomForestClassifier' 是我們給這個模型系列取的名稱
    mlflow.sklearn.log_model(
        sk_model=model_v1,
        artifact_path="iris_rf_model",
        registered_model_name="IrisRandomForestClassifier" 
    )
    
    run_id_v1 = run.info.run_id
    print(f"模型版本 1 訓練完成！Accuracy: {accuracy_v1:.4f}")
    print(f"MLflow Run ID: {run_id_v1}")
    print(f"模型已註冊為 'IrisRandomForestClassifier' 的版本 1。")


# --- 第二次訓練模型 (版本 2)，假設我們調整了參數 ---
print("\n--- 訓練模型版本 2 ---")
with mlflow.start_run(run_name="Iris_Classifier_V2") as run:
    # 定義模型參數 (調整 n_estimators)
    n_estimators_v2 = 150 # 增加了樹的數量
    max_depth_v2 = 10
    random_state_v2 = 42

    # 訓練模型
    model_v2 = RandomForestClassifier(
        n_estimators=n_estimators_v2,
        max_depth=max_depth_v2,
        random_state=random_state_v2
    )
    model_v2.fit(X_train, y_train)

    # 預測與評估
    y_pred_v2 = model_v2.predict(X_test)
    accuracy_v2 = accuracy_score(y_test, y_pred_v2)
    
    # Log 參數到 MLflow
    mlflow.log_param("n_estimators", n_estimators_v2)
    mlflow.log_param("max_depth", max_depth_v2)
    mlflow.log_param("random_state", random_state_v2)
    
    # Log 指標到 MLflow
    mlflow.log_metric("accuracy", accuracy_v2)
    
    # 再次將模型儲存並註冊到 MLflow Model Registry
    # 注意：使用相同的 registered_model_name 會自動創建一個新版本！
    mlflow.sklearn.log_model(
        sk_model=model_v2,
        artifact_path="iris_rf_model",
        registered_model_name="IrisRandomForestClassifier" 
    )
    
    run_id_v2 = run.info.run_id
    print(f"模型版本 2 訓練完成！Accuracy: {accuracy_v2:.4f}")
    print(f"MLflow Run ID: {run_id_v2}")
    print(f"模型已註冊為 'IrisRandomForestClassifier' 的版本 2。")

print("\n模型訓練和註冊已完成。現在請在終端機執行 'mlflow ui' 並在瀏覽器中查看 MLflow UI。")
print("你可以在 'Models' 頁面看到 'IrisRandomForestClassifier' 模型的不同版本。")
```

### 執行與查看結果

1.  **開啟終端機 (Terminal)**，進入你存放這個 Python 檔案的目錄。
2.  **執行 MLflow UI**：
    ```bash
    mlflow ui
    ```
    這會在你的瀏覽器中開啟一個介面 (通常是 `http://localhost:5000`)。
3.  **執行 Python 程式碼**：在另一個終端機視窗中，執行你的 Python 檔案：
    ```bash
    python your_script_name.py
    ```

當程式碼執行完成後，回到你的 MLflow UI 介面：

*   **Experiments (實驗)** 頁面：你會看到兩次 Run (Iris_Classifier_V1 和 Iris_Classifier_V2)，點擊進去可以看到各自的參數、指標和儲存的模型檔案。
*   **Models (模型)** 頁面：你會看到一個名為 `IrisRandomForestClassifier` 的模型。點擊它，你就能看到這個模型的所有版本 (Version 1, Version 2)。你可以比較它們的指標、查看是哪個 Run 產生了它們，甚至可以將它們的 "Stage" 從 `None` 轉換為 `Staging` (預備環境) 或 `Production` (生產環境)，這就是模型管理的核心！

### 模型階段 (Model Stages)

在 MLflow Model Registry 中，你可以為每個模型版本設定不同的階段，這對於管理模型生命週期至關重要：

*   **None (無)**：預設狀態，表示模型尚未經過審核或分配特定用途。
*   **Staging (預備)**：表示模型正在測試、審核，可能在部署到生產環境前進行進一步驗證。
*   **Production (生產)**：表示模型已經通過所有測試，被認為是穩定可靠的，正在或即將在實際應用中使用。
*   **Archived (已歸檔)**：表示模型不再使用，但仍保留歷史記錄以供參考。

你可以在 MLflow UI 中手動將模型版本從一個階段轉換到另一個階段。這是一個非常強大的功能，讓你的團隊能夠有條不紊地管理模型部署流程。

### 小結與鼓勵

恭喜你！今天你不僅訓練了模型，更學會了如何利用 MLflow 這樣專業的工具來對你的模型進行版本控制和管理。這是一個巨大的里程碑，因為它將你的機器學習專案從單純的實驗提升到了更具工程化、更具可重現性和可擴展性的 MLOps 層次。

現在，你已經有了追蹤不同模型版本、比較它們效能、甚至管理它們生命週期的能力。這對於未來的專案和團隊協作來說，是至關重要的技能。繼續探索 MLflow 的更多功能吧，你會發現 MLOps 的世界充滿了無限可能！

明天，我們將繼續深入 MLOps 的其他精彩主題！保持好奇心，持續學習！加油！