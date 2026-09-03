哈囉，我的程式小夥伴！👋 歡迎來到我們的【第 123 天】！你已經在機器學習的旅程中走了這麼遠，從資料清理、模型訓練，到評估，相信你已經能打造出很棒的模型了。

不過，你可能會發現，當模型需要不斷更新、部署、監控時，手動操作會變得非常繁瑣且容易出錯。這時候，我們就需要請出今天的主角：**MLOps 工作流程編排與自動化！**

別擔心，「MLOps」聽起來可能有點高大上，但它其實就像是為你的機器學習專案搭建一條自動化的生產線，讓你的模型從「實驗室」走向「真實世界」，而且能穩定、有效率地運行。今天，我們要一起來實作一個簡化的 MLOps 工作流程，讓你親身體驗自動化的魅力！

---

## 主題：【第 123 天：實戰：MLOps 工作流程編排與自動化】

### 🎯 為什麼是 MLOps？

想像一下，你的模型需要每週重新訓練一次，以適應新的資料趨勢。如果你每次都要手動執行資料處理、訓練、評估、部署這些步驟，是不是想想就覺得頭大？MLOps 的目標就是讓這些重複的步驟自動化、可重複、可追蹤，大幅提升效率並減少人為錯誤。

我們今天的目標是：建立一個簡單的 Python 腳本，它能自動依序執行以下步驟：
1.  **資料準備 (Data Preparation)**
2.  **模型訓練 (Model Training)**
3.  **模型評估 (Model Evaluation)**
4.  **模型部署 (Model Deployment - 簡化版)**

### 🛠️ 我們的工具：Python `subprocess` 模組

在真實世界中，你會用到像 Airflow、Kubeflow 或 MLflow Pipelines 這樣的專業工具來編排工作流程。但作為初學者，我們可以利用 Python 內建的 `subprocess` 模組來模擬這個過程，它能讓我們在一個 Python 腳本中呼叫並執行其他的 Python 腳本或命令。

---

### 📝 實作：建立我們的 MLOps 工作流程

我們將建立四個獨立的 Python 腳本，分別負責一個步驟，然後再寫一個「主控」腳本來依序執行它們。

#### 步驟 1: 準備資料 (`01_data_prep.py`)

這個腳本會生成一些假資料，並將其保存起來，模擬資料準備的過程。

```python
# 01_data_prep.py
import pandas as pd
import numpy as np
import os

def prepare_data():
    print("--- 步驟 1: 正在準備資料 ---")
    # 模擬生成一些假資料
    np.random.seed(42)
    data = {
        'feature_1': np.random.rand(100) * 10,
        'feature_2': np.random.rand(100) * 5,
        'target': np.random.randint(0, 2, 100) # 0 或 1
    }
    df = pd.DataFrame(data)

    # 創建一個資料夾來存放我們的產出
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    # 將資料保存為 CSV
    data_path = 'artifacts/prepared_data.csv'
    df.to_csv(data_path, index=False)
    print(f"資料準備完成！已保存至 {data_path}")
    return data_path

if __name__ == "__main__":
    prepare_data()
```

#### 步驟 2: 訓練模型 (`02_train_model.py`)

這個腳本會載入準備好的資料，訓練一個簡單的模型，並將其保存。

```python
# 02_train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib # 用於保存/載入模型
import os

def train_model():
    print("--- 步驟 2: 正在訓練模型 ---")
    data_path = 'artifacts/prepared_data.csv'
    if not os.path.exists(data_path):
        print(f"錯誤：找不到資料文件 {data_path}。請先執行資料準備。")
        exit(1)

    df = pd.read_csv(data_path)

    X = df[['feature_1', 'feature_2']]
    y = df['target']

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    model_path = 'artifacts/trained_model.pkl'
    joblib.dump(model, model_path)
    print(f"模型訓練完成！已保存至 {model_path}")
    return model_path

if __name__ == "__main__":
    train_model()
```

#### 步驟 3: 評估模型 (`03_evaluate_model.py`)

這個腳本會載入訓練好的模型和資料，進行評估，並輸出一個分數。

```python
# 03_evaluate_model.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import os

def evaluate_model():
    print("--- 步驟 3: 正在評估模型 ---")
    data_path = 'artifacts/prepared_data.csv'
    model_path = 'artifacts/trained_model.pkl'

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print(f"錯誤：找不到資料或模型文件。請確保已完成前續步驟。")
        exit(1)

    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X = df[['feature_1', 'feature_2']]
    y_true = df['target']
    y_pred = model.predict(X)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"模型評估完成！準確度 (Accuracy): {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    evaluate_model()
```

#### 步驟 4: 部署模型 (`04_deploy_model.py`)

這個腳本會模擬部署的動作。在真實世界中，這可能涉及到將模型上傳到雲端服務或設定 API 端點。

```python
# 04_deploy_model.py
import os

def deploy_model():
    print("--- 步驟 4: 正在部署模型 ---")
    model_path = 'artifacts/trained_model.pkl'
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型文件 {model_path}。請確保模型已訓練。")
        exit(1)

    # 這裡我們只是一個模擬，印出訊息
    print(f"模型 '{model_path}' 已模擬部署完成！")
    print("在真實世界中，這可能會涉及到：")
    print(" - 將模型上傳到雲端服務 (AWS Sagemaker, Google AI Platform, Azure ML)")
    print(" - 建立 REST API 端點供應用程式使用")
    print(" - 更新模型註冊表 (Model Registry)")

if __name__ == "__main__":
    deploy_model()
```

#### 步驟 5: 編排器 (`orchestrate_mlops.py`)

這是我們的核心，它將負責依序觸發上述四個腳本。

```python
# orchestrate_mlops.py
import subprocess
import sys # 用於獲取 Python 解釋器的路徑

def run_script(script_name):
    """
    執行指定的 Python 腳本。
    """
    print(f"\n===== 開始執行 {script_name} =====")
    # 使用 sys.executable 確保我們使用當前環境的 Python 解釋器
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"！！！ 執行 {script_name} 失敗 ！！！")
        print("標準輸出：")
        print(result.stdout)
        print("錯誤輸出：")
        print(result.stderr)
        return False
    else:
        print(f"執行 {script_name} 成功！")
        print("輸出：")
        print(result.stdout)
        return True

def main_mlops_workflow():
    print("====== MLOps 自動化工作流程啟動 ======\n")

    # 定義我們的工作流程步驟
    workflow_steps = [
        "01_data_prep.py",
        "02_train_model.py",
        "03_evaluate_model.py",
        "04_deploy_model.py"
    ]

    for step in workflow_steps:
        if not run_script(step):
            print(f"\n！！！ 工作流程在 {step} 處中斷。請檢查錯誤訊息。")
            break
    else: # 如果所有步驟都成功執行
        print("\n====== MLOps 自動化工作流程成功完成！恭喜你！ ======")

if __name__ == "__main__":
    main_mlops_workflow()
```

---

### 🚀 運行你的自動化工作流程

1.  **創建文件夾：** 在你的專案根目錄下，創建一個名為 `artifacts` 的文件夾。
2.  **保存腳本：** 將上述五個腳本保存到同一個目錄中。
3.  **打開終端機 (Terminal / Command Prompt)：**
4.  **執行主控腳本：**
    ```bash
    python orchestrate_mlops.py
    ```

你將會看到程式依序執行每個步驟，印出訊息，是不是很酷呢？你的機器學習模型現在有了自己的「自動化生產線」！

---

### 🌟 接下來呢？

這個例子雖然簡化，但它展示了 MLOps 自動化的核心思想：將複雜的流程拆解成可獨立運行的模組，再用一個主控器將它們串聯起來。

在真實世界的 MLOps 中，你還會考慮：

*   **版本控制：** 不僅是程式碼，資料和訓練好的模型也需要版本控制 (例如：DVC)。
*   **參數管理：** 追蹤不同模型訓練的超參數、指標 (例如：MLflow)。
*   **持續整合/持續部署 (CI/CD)：** 當程式碼或資料有變動時，自動觸發工作流程。
*   **監控：** 部署後的模型表現如何？是否有資料偏移？需要重新訓練嗎？
*   **錯誤處理與通知：** 當工作流程失敗時，如何通知相關人員。

---

### 🎉 恭喜你！

你看，是不是很棒？從手動執行到現在的一鍵啟動，你已經跨出了 MLOps 的第一步！這只是一個開端，但它為你打開了通往更高效、更可靠的機器學習系統的大門。繼續探索，你會發現 MLOps 的世界充滿了無限可能！

保持好奇，持續學習！我們下一次見！🚀