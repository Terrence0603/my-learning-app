好的，各位同學！歡迎來到我們 MLOps 系列的第 112 天！

## 【第 112 天：實戰：MLOps 平台整合與自動化工作流建構】

嘿，各位未來的 MLOps 大師們！

還記得我們前幾天學習了如何訓練模型、評估模型嗎？這些都是機器學習的核心。但如果你想讓你的模型不僅僅停留在你的筆電上完美運行，而是能**自動化、可靠地部署、監控，並不斷迭代更新**，那麼今天這一課將是你的超級英雄披風！

想像一下，你的模型訓練、評估、甚至註冊到模型庫的過程，不再需要你手動點擊、運行腳本，而是像施展魔法一樣，當你把新代碼推送到程式碼倉庫時，它就自動完成了！這就是我們今天要探索的 MLOps 平台整合與自動化工作流的魔力！

別擔心，我們不會一下子深入到超複雜的雲端架構。今天，我們將聚焦在兩個對初學者非常友善且強大的工具：
1.  **MLflow**：作為你的實驗追蹤器和模型註冊中心。
2.  **GitHub Actions**：作為你的自動化機器人，幫你執行工作流。

我們的目標很簡單：**當我們將模型訓練程式碼推送到 GitHub 時，GitHub Actions 會自動執行訓練，並將實驗結果（參數、指標）和訓練好的模型記錄到 MLflow！**

---

### MLOps 核心概念複習 (Why we need this?)

在進入實作之前，讓我們快速回顧一下為什麼 MLOps 如此重要：

*   **自動化 (Automation)**：減少手動操作，降低錯誤率，提高效率。
*   **可重現性 (Reproducibility)**：確保每次訓練或部署的結果都是可驗證的。
*   **版本控制 (Versioning)**：對代碼、數據和模型進行版本管理。
*   **監控 (Monitoring)**：部署後模型的性能監控。

今天，我們主要聚焦在**自動化**和**可重現性**上，利用 MLflow 來記錄，並用 GitHub Actions 來自動執行。

---

### 環境準備 (起手式)

首先，你需要一個 Python 環境，並安裝必要的套件：

```bash
pip install mlflow scikit-learn pandas
```

然後，在你的專案根目錄下，建立一個 `requirements.txt` 檔案，內容如下：

```
mlflow
scikit-learn
pandas
```

---

### 步驟一：整合 MLflow 到你的模型訓練程式碼 (`train.py`)

我們來寫一個簡單的模型訓練腳本 `train.py`，它會模擬一個數據集，訓練一個迴歸模型，並將訓練過程中的參數、性能指標和最終的模型都記錄到 MLflow。

```python
# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設定 MLflow 追蹤伺服器 (如果沒有特別設定，預設會記錄到 ./mlruns)
# 在實際應用中，你可能會設定為遠端伺服器:
# mlflow.set_tracking_uri("http://localhost:5000") # 或其他遠端 URI

def train_model(alpha=0.5, l1_ratio=0.5):
    """
    訓練一個簡單的線性迴歸模型，並將結果記錄到 MLflow。
    """
    with mlflow.start_run():
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

        # 1. 準備數據 (這裡我們用模擬數據)
        data = {
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'target': [12, 11, 13, 15, 14, 16, 18, 17, 19, 21]
        }
        df = pd.DataFrame(data)

        X = df[['feature1', 'feature2']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. 訓練模型 (這裡使用簡單的線性迴歸)
        model = LinearRegression() # 為了簡單，這裡不使用 Ridge 或 Lasso，直接用 LinearRegression
        model.fit(X_train, y_train)

        # 3. 評估模型
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        logger.info(f"模型訓練完成 - RMSE: {rmse:.2f}, R2: {r2:.2f}")

        # 4. 使用 MLflow 記錄參數、指標和模型
        mlflow.log_param("alpha", alpha) # 即使模型沒用到，也可以示範記錄參數
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("model_type", "LinearRegression") # 記錄模型類型

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(model, "model") # 將模型保存為 "model" 這個 artifacts

        # 你也可以註冊模型到 MLflow Model Registry
        # mlflow.register_model(
        #     model_uri="runs:/{}/model".format(mlflow.active_run().info.run_id),
        #     name="SimpleLinearRegressionModel"
        # )
        # logger.info("模型已註冊到 Model Registry")

if __name__ == "__main__":
    train_model()
```

### 步驟二：建構 GitHub Actions 自動化工作流 (`.github/workflows/mlops.yml`)

現在，我們來建立 GitHub Actions 的配置文件。在你的專案根目錄下，建立 `.github/workflows/` 資料夾，然後在裡面建立 `mlops.yml` 檔案：

```yaml
# .github/workflows/mlops.yml
name: MLOps_Workflow

on:
  push:
    branches:
      - main # 當有代碼推送到 main 分支時觸發此工作流

jobs:
  train_and_log_model:
    runs-on: ubuntu-latest # 在最新的 Ubuntu 虛擬機上運行

    steps:
      - name: 檢查程式碼
        uses: actions/checkout@v3 # 獲取你的 GitHub 倉庫程式碼

      - name: 設定 Python 環境
        uses: actions/setup-python@v4
        with:
          python-version: '3.9' # 指定 Python 版本

      - name: 安裝依賴套件
        run: |
          pip install -r requirements.txt # 安裝你在 requirements.txt 中列出的套件

      - name: 運行模型訓練與 MLflow 記錄
        run: |
          # 設定 MLflow 追蹤 URI
          # 在 GitHub Actions 中，如果沒有設定遠端 MLflow 伺服器，
          # MLflow 會在當前工作目錄下建立 'mlruns' 資料夾來儲存日誌。
          # 如果你有自己的遠端 MLflow 伺服器，可以在這裡設定 MLFLOW_TRACKING_URI
          # 例如: export MLFLOW_TRACKING_URI="https://your-mlflow-server.com"
          python train.py
        env:
          # 如果你有 MLflow 遠端追蹤伺服器，這裡可以設定認證資訊
          # MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_USER }}
          # MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_PASSWORD }}
          # 或者對於雲端服務，如 Azure ML:
          # AZURE_ML_RESOURCE_GROUP: ${{ secrets.AZURE_ML_RESOURCE_GROUP }}
          # AZURE_ML_WORKSPACE_NAME: ${{ secrets.AZURE_ML_WORKSPACE_NAME }}
          # ...等等
          # 對於本例，我們假設是本地 mlruns
          MLFLOW_TRACKING_URI: "." # 設定 MLflow 追蹤到當前目錄 (會在 runner 上產生 mlruns 資料夾)
```

---

### 步驟三：推送到 GitHub，見證魔法！

1.  **初始化 Git 倉庫 (如果還沒做的話)**：
    ```bash
    git init
    git add .
    git commit -m "Initial commit with MLOps workflow"
    git branch -M main
    git remote add origin https://github.com/你的用戶名/你的倉庫名.git
    git push -u origin main
    ```
2.  **上傳程式碼**：將 `train.py`, `requirements.txt`, 和 `.github/workflows/mlops.yml` 這三個檔案添加到你的 Git 倉庫並推送到 `main` 分支。
    ```bash
    git add .
    git commit -m "Add MLflow integration and GitHub Actions workflow"
    git push origin main
    ```

一旦你推送到 `main` 分支，GitHub Actions 就會被觸發！你可以到你的 GitHub 倉庫頁面，點擊 "Actions" 選項卡，你就會看到你的 `MLOps_Workflow` 正在運行！

當工作流成功完成後：
*   在 GitHub Actions 的日誌中，你會看到 `train.py` 的運行輸出。
*   MLflow 會在 GitHub Actions 的運行環境中創建一個 `mlruns` 資料夾，並記錄所有實驗數據。**（請注意：這個 `mlruns` 資料夾會隨著 GitHub Actions runner 的終止而消失。在真實世界中，你會配置一個遠端的 MLflow 追蹤伺服器，讓這些數據能夠持久化保存並集中管理。）**

---

### 恭喜你，你已經是 MLOps 的小小建築師了！

今天，我們跨出了 MLOps 的重要一步：將 MLflow 實驗追蹤與 GitHub Actions 自動化工作流結合。你現在已經能夠：

*   使用 MLflow 記錄你的模型訓練參數、指標和模型本身。
*   利用 GitHub Actions 自動執行你的 ML 任務，無需手動介入。

這只是 MLOps 世界的冰山一角。未來，我們還可以探索如何將模型自動部署到生產環境、如何監控模型性能、如何處理數據漂移等等。但今天，你已經掌握了自動化 ML 工作流的基石！

繼續保持這份好奇心和實作精神，你會在 MLOps 的道路上越走越遠！

下一堂課見！祝你編程愉快！