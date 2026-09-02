哈囉，未來的 MLOps 大師！恭喜你來到我們學習旅程的第 122 天！

走到這一步，你肯定已經對機器學習的各個環節駕輕就熟了。今天我們要將你的技能提升到另一個層次，進入 MLOps (Machine Learning Operations) 的核心地帶：**持續整合與持續部署 (CI/CD)**。別擔心，聽起來可能有點複雜，但我們會用最輕鬆、最實用的方式來理解它，讓你的 ML 專案從此告別手動部署的煩惱，變得更穩定、更有效率！

---

## 【第 122 天：實戰：MLOps 持續整合與持續部署 (CI/CD)】

### 🚀 踏上自動化之路：CI/CD 在 MLOps 中的魔力

想像一下，你和團隊辛辛苦苦開發了一個超棒的機器學習模型。每次有新的資料進來，或是程式碼做了小小的調整，你都得手動跑一次模型訓練、評估，然後小心翼翼地更新部署。是不是聽起來就很累人，而且還容易出錯？

這就是 CI/CD 要來解救你的地方！

*   **持續整合 (Continuous Integration, CI)**：簡單來說，就是團隊成員把程式碼整合到主分支後，系統會**自動**進行一系列的檢查（例如：程式碼風格檢查、單元測試、資料驗證）。這樣可以確保你的新程式碼不會破壞既有的功能，並及早發現問題。
*   **持續部署 (Continuous Deployment, CD)**：當 CI 的所有檢查都通過後，系統會**自動**把你的模型或應用部署到測試環境、預演環境，甚至是生產環境。這意味著你可以更快、更安全地將新模型或功能交付給使用者。

在 MLOps 中，CI/CD 不僅僅是針對程式碼，它還要關心：
1.  **程式碼**：模型訓練腳本、預處理程式碼、API 程式碼。
2.  **資料**：新數據的驗證、資料漂移的監測。
3.  **模型**：新模型的訓練、評估、版本管理。

我們的目標是建立一個「當你推送到 Git 倉庫時，所有檢查、訓練、評估都能自動完成」的流程，這會大大提升開發效率和模型品質！

### 💡 我們的 MLOps CI/CD 流程簡化版

今天，我們將以一個簡化的範例，使用 **GitHub Actions** 來實現 CI/CD。當你把程式碼推送到 GitHub 倉庫的 `main` 分支時，它將會自動執行以下步驟：

1.  **程式碼檢出 (Checkout Code)**：獲取最新的程式碼。
2.  **設定 Python 環境 (Set up Python)**：準備好執行 Python 腳本所需的環境。
3.  **安裝依賴 (Install Dependencies)**：安裝 `requirements.txt` 中定義的所有套件。
4.  **程式碼風格檢查 (Linting)**：使用 `flake8` 檢查程式碼風格，確保團隊的程式碼風格一致。
5.  **單元測試 (Unit Tests)**：運行 `pytest` 來驗證你的程式碼邏輯是否正確。
6.  **模擬模型訓練 (Simulate Model Training)**：執行一個腳本，模擬模型的訓練過程並保存模型。
7.  **模擬模型評估 (Simulate Model Evaluation)**：執行一個腳本，載入訓練好的模型並進行評估，檢查模型性能是否符合預期。

### ✍️ 動手實作：GitHub Actions 範例

首先，你需要一個 GitHub 倉庫。在你的專案根目錄下，我們來創建以下這些檔案：

#### `requirements.txt`

```
scikit-learn==1.3.0
pandas==2.0.0
pytest==7.4.0
flake8==6.1.0
joblib==1.3.2
```

#### `src/train.py` (模型訓練腳本)

```python
# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

print("--- 正在執行模型訓練 ---")

# 模擬資料生成
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'target': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df[['feature1', 'feature2']]
y = df['target']

# 訓練模型
model = LogisticRegression()
model.fit(X, y)

# 保存模型
os.makedirs('models', exist_ok=True) # 確保 models 資料夾存在
model_path = 'models/model.pkl'
joblib.dump(model, model_path)

print(f"模型已訓練並保存至 {model_path}")
print("--- 模型訓練完成 ---")
```

#### `src/evaluate.py` (模型評估腳本)

```python
# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os

print("--- 正在執行模型評估 ---")

# 載入模型
model_path = 'models/model.pkl'
if not os.path.exists(model_path):
    print(f"錯誤：找不到模型檔 {model_path}，請確保訓練腳本已執行。")
    exit(1)

model = joblib.load(model_path)
print(f"模型已從 {model_path} 載入。")

# 模擬評估資料
eval_data = {
    'feature1': [1, 2, 9, 10],
    'feature2': [10, 9, 2, 1],
    'target': [0, 0, 1, 1]
}
eval_df = pd.DataFrame(eval_data)

X_eval = eval_df[['feature1', 'feature2']]
y_true = eval_df['target']

y_pred = model.predict(X_eval)
accuracy = accuracy_score(y_true, y_pred)

print(f"模型在評估資料上的準確度 (Accuracy): {accuracy:.2f}")

# 這裡可以加入更多評估指標或閾值判斷
if accuracy < 0.7:
    print("警告：模型準確度低於預期！")

print("--- 模型評估完成 ---")
```

#### `tests/test_model.py` (單元測試腳本)

```python
# tests/test_model.py
import os
import joblib
import pytest

# 假設模型訓練後會生成一個 'models/model.pkl' 檔案
# 這個測試檢查該檔案是否存在
def test_model_artifact_exists():
    """
    測試模型訓練後是否成功產生 model.pkl 檔案。
    """
    # 這裡我們需要先執行一次訓練腳本，讓它產生模型檔
    # 在 GitHub Actions 中，train.py 會在測試之前運行
    # 所以這個測試是確認檔案是否真的被產生了
    model_path = 'models/model.pkl'
    assert os.path.exists(model_path), f"模型檔案 {model_path} 不存在！"

def test_model_can_be_loaded():
    """
    測試訓練好的模型是否可以成功載入。
    """
    model_path = 'models/model.pkl'
    try:
        model = joblib.load(model_path)
        assert model is not None, "載入的模型為 None！"
        assert hasattr(model, 'predict'), "載入的物件不像是 Scikit-learn 模型 (缺少 predict 方法)。"
    except Exception as e:
        pytest.fail(f"載入模型失敗：{e}")

# 你可以根據你的模型邏輯，添加更多單元測試
# 例如，測試模型的預測結果是否符合預期區間
```

#### `.github/workflows/mlops_ci.yml` (GitHub Actions 工作流配置)

這是我們 CI/CD 的核心！在 `.github/workflows/` 目錄下創建此文件：

```yaml
name: MLOps CI/CD Pipeline

on:
  push:
    branches:
      - main # 當有程式碼推送到 main 分支時，觸發此工作流

jobs:
  build-and-test:
    runs-on: ubuntu-latest # 在 Ubuntu 環境中運行

    steps:
      - name: Checkout code
        uses: actions/checkout@v3 # 檢出你的 Git 倉庫程式碼

      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: '3.9' # 設定 Python 版本

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip # 更新 pip
          pip install -r requirements.txt # 安裝所有依賴套件

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics # 執行 flake8 程式碼風格檢查
          # E9, F63, F7, F82 是常見的錯誤和語法警告

      - name: Run unit tests
        run: pytest tests/ # 執行 tests/ 目錄下的所有單元測試

      - name: Simulate Model Training
        run: python src/train.py # 執行模型訓練腳本

      - name: Simulate Model Evaluation
        run: python src/evaluate.py # 執行模型評估腳本
```

### 🧐 解讀 `mlops_ci.yml`

*   `name`: 這個工作流的名稱，會顯示在 GitHub Actions 頁面。
*   `on: push: branches: - main`: 這告訴 GitHub Actions，當有人將程式碼 `push` 到 `main` 分支時，就觸發這個工作流。
*   `jobs: build-and-test`: 定義一個名為 `build-and-test` 的工作 (Job)。一個工作流可以有多個工作。
*   `runs-on: ubuntu-latest`: 指定這個工作會在什麼操作系統環境中運行。
*   `steps`: 這是工作的核心，一系列依序執行的動作。
    *   `uses: actions/checkout@v3`：這是一個 GitHub Action，用來把你的倉庫程式碼下載到運行環境中。
    *   `uses: actions/setup-python@v4`：另一個 GitHub Action，用來設定特定版本的 Python。
    *   `name: Install dependencies` 及 `run:`：`name` 是這個步驟的名稱，`run` 則是要執行的 shell 命令。這裡我們安裝了 `requirements.txt` 中的所有套件。
    *   `name: Lint with flake8`：執行 `flake8` 檢查程式碼品質。
    *   `name: Run unit tests`：執行 `pytest` 運行單元測試。
    *   `name: Simulate Model Training` 和 `name: Simulate Model Evaluation`：執行我們前面寫好的 Python 腳本，模擬訓練和評估過程。

### 🎉 試運行你的第一個 MLOps CI/CD！

1.  將上述所有檔案組織到你的 GitHub 倉庫中，確保檔案路徑正確（特別是 `.github/workflows/mlops_ci.yml`）。
2.  將這些檔案提交 (commit) 並推送到 (push) 你的 GitHub 倉庫的 `main` 分支。
3.  打開你的 GitHub 倉庫頁面，點擊上方的 "Actions" 選項卡。你應該會看到一個新的工作流正在運行！
4.  點擊進入，你可以看到每個步驟的詳細日誌輸出。如果一切順利，所有步驟都會顯示綠色的勾勾！

---

### 結語：展望未來

恭喜你！今天你已經邁出了 MLOps CI/CD 的第一步。這只是一個簡單的範例，但在真實世界中，你可以擴展它：

*   **更完善的資料驗證**：在訓練前檢查資料品質。
*   **模型註冊 (Model Registry)**：將訓練好的模型及元數據（例如準確度、F1 Score）自動記錄到 MLflow 或其他模型註冊中心。
*   **A/B 測試部署**：自動將新模型部署到一小部分用戶進行測試。
*   **回滾機制**：當新模型表現不佳時，自動切換回舊模型。

CI/CD 是 MLOps 的基石，它讓你和你的團隊能夠更快速、更可靠地迭代和部署機器學習模型。請記住，每一次自動化的成功運行，都是你 MLOps 之路上的一個小小勝利！

堅持下去，你一定會成為一位卓越的 MLOps 專家！我們下個主題再見！