哈囉！第 166 天的你，是不是感覺離程式大師又更近一步了呢？恭喜你！今天我們要來挑戰一個讓機器學習模型真正「活起來」的關鍵技能——MLOps 自動化模型再訓練與部署管線！

你可能已經學會了如何訓練出一個很棒的模型。但模型就像是植物，它需要定期澆水、施肥（也就是用新數據再訓練），才能保持生機勃勃。而且當模型更新了，我們還需要能快速、穩定地把它部署到實際應用中。這就是 MLOps 的魅力所在！

### 🚀 MLOps 是什麼？為什麼要自動化？

你可以想像 MLOps 就是把軟體工程的「維運 (Operations)」精神帶入機器學習 (Machine Learning) 的世界。它關注的重點是如何讓模型的開發、部署、監控和維護變得更有效率、更可靠。

**為什麼要自動化呢？**

1.  **模型漂移 (Model Drift)：** 數據會隨著時間變化，你的模型可能會因此「過時」，效能下降。自動化再訓練能確保你的模型總是能學習到最新的趨勢。
2.  **效率提升：** 手動重新訓練、評估、部署模型費時費力，且容易出錯。自動化管線能讓你一鍵完成這些繁瑣的步驟。
3.  **快速迭代：** 當有新數據、新想法時，你可以更快地測試、部署新的模型版本。

### ⚙️ 自動化管線的核心概念

一個 MLOps 自動化管線通常包含以下幾個步驟：

1.  **觸發 (Trigger)：** 當有新數據、時間排程到、或是開發者手動觸發時，管線就會啟動。
2.  **數據準備 (Data Preparation)：** 自動收集、清洗、處理最新的數據。
3.  **模型訓練 (Model Training)：** 使用新數據自動重新訓練模型。
4.  **模型評估 (Model Evaluation)：** 評估新模型的效能，並與現有（生產環境）模型進行比較。
5.  **模型部署 (Model Deployment)：** 如果新模型表現更好，則自動替換掉生產環境中的舊模型。
6.  **模型監控 (Model Monitoring)：** 持續監控生產環境中模型的效能和數據變化。

今天，我們將聚焦在「訓練」、「評估」和「部署」這三個核心步驟的自動化，並透過一個簡單的 Python 腳本來模擬整個流程。

### 📝 程式碼範例：讓模型動起來！

為了讓初學者更容易理解，我們將用一個 Python 腳本來模擬一個簡單的 MLOps 管線。這個腳本將會：

1.  模擬生成新數據並訓練一個新的模型。
2.  將新模型的準確率與當前「生產環境」的模型進行比較。
3.  如果新模型表現更好，則將其「部署」到生產環境（這裡指的是替換掉一個特定的模型檔案）。

**檔案結構建議：**

```
mlops_pipeline/
├── pipeline.py
└── models/                  # 存放模型的資料夾
    ├── production_model.pkl # 當前生產環境模型
    └── production_accuracy.txt # 當前生產環境模型的準確率
```

**`pipeline.py`**

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# 定義模型存放的路徑
MODEL_DIR = "models"
PROD_MODEL_PATH = os.path.join(MODEL_DIR, "production_model.pkl")
PROD_ACCURACY_PATH = os.path.join(MODEL_DIR, "production_accuracy.txt")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "new_trained_model.pkl")
NEW_ACCURACY_PATH = os.path.join(MODEL_DIR, "new_accuracy.txt")

# 確保模型資料夾存在
os.makedirs(MODEL_DIR, exist_ok=True)

def train_and_evaluate_new_model():
    """
    模擬收集新數據並訓練模型，然後評估其性能。
    """
    print("--- 步驟 1: 收集新數據並訓練模型 ---")
    
    # 模擬生成新數據（每次運行會略有不同，模擬數據變化）
    # random_state 加上隨機數，使得每次生成的數據略有不同，模擬真實世界的數據漂移
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, 
                               random_state=42 + np.random.randint(0, 100)) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✔️ 新模型訓練完成，準確率: {accuracy:.4f}")

    # 保存新模型及其準確率
    joblib.dump(model, NEW_MODEL_PATH)
    with open(NEW_ACCURACY_PATH, "w") as f:
        f.write(str(accuracy))
    return accuracy

def get_current_production_info():
    """
    獲取目前生產環境中模型的準確率。
    """
    if os.path.exists(PROD_MODEL_PATH) and os.path.exists(PROD_ACCURACY_PATH):
        try:
            with open(PROD_ACCURACY_PATH, "r") as f:
                prod_accuracy = float(f.read())
            print(f"目前生產環境模型準確率: {prod_accuracy:.4f}")
            return prod_accuracy
        except ValueError: # 防止文件內容損壞
            print("❌ 無法讀取生產環境準確率，視為 0.0。")
            return 0.0
    print("目前無生產環境模型或準確率資訊。")
    return 0.0 # 如果沒有生產模型，則視為準確率為 0.0

def deploy_model(new_accuracy):
    """
    根據新模型的準確率決定是否部署。
    """
    print("\n--- 步驟 2: 評估並決定是否部署新模型 ---")
    prod_accuracy = get_current_production_info()

    if new_accuracy > prod_accuracy:
        print(f"✨ 新模型 ({new_accuracy:.4f}) 優於現有模型 ({prod_accuracy:.4f})，準備部署！")
        # 將新模型複製到生產環境路徑
        joblib.dump(joblib.load(NEW_MODEL_PATH), PROD_MODEL_PATH) 
        # 更新生產環境的準確率記錄
        with open(PROD_ACCURACY_PATH, "w") as f:
            f.write(str(new_accuracy))
        print("✅ 新模型已成功部署到生產環境！")
    else:
        print(f"😕 新模型 ({new_accuracy:.4f}) 未優於現有模型 ({prod_accuracy:.4f})。維持現有模型。")

if __name__ == "__main__":
    print("🚀 MLOps 自動化模型再訓練與部署管線啟動！")
    new_model_accuracy = train_and_evaluate_new_model()
    deploy_model(new_model_accuracy)
    print("\n✅ 管線執行完畢。")
    print("你可以嘗試多次執行 `python pipeline.py`，觀察模型是否會因為準確率提升而被重新部署！")

```

#### 如何執行？

1.  將上述程式碼儲存為 `pipeline.py`。
2.  在同一個資料夾下，確保有一個名為 `models` 的空資料夾。
3.  打開你的終端機或命令提示字元，導航到 `mlops_pipeline` 資料夾。
4.  執行指令：
    ```bash
    python pipeline.py
    ```

**試著執行幾次看看！**

你會發現，第一次執行時，新模型會被部署。之後每次執行，因為我們模擬了數據的輕微變化，所以新模型的準確率會略有不同。只有當新模型比當前「生產環境」的模型表現更好時，它才會被替換掉。這就模擬了一個基礎的自動化再訓練與部署管線！

### 🎯 下一步呢？

今天的範例是簡化過的，但它展示了 MLOps 的核心邏輯。在真實世界中，你會搭配更多專業工具來實現：

*   **版本控制 (Git/GitHub)：** 管理你的程式碼和管線定義。
*   **容器化 (Docker)：** 將你的模型和所有依賴打包成一個獨立的運行環境，確保在任何地方都能一致運行。
*   **CI/CD 工具 (GitHub Actions, GitLab CI/CD, Jenkins)：** 自動觸發並執行你的 `pipeline.py` 腳本。
*   **數據版本控制 (DVC)：** 管理你的數據集版本。
*   **模型註冊中心 (MLflow Model Registry, Sagemaker MLOps)：** 追蹤、管理和版本化你的模型。
*   **模型服務化 (Flask, FastAPI, TensorFlow Serving, Sagemaker Endpoint)：** 將部署好的模型以 API 服務的形式提供給應用程式使用。

### 🎉 結語

恭喜你，又掌握了一項 MLOps 的核心技能！從今天開始，你的機器學習模型不再是靜態的藝術品，而是能隨著數據變化而成長、進化的小生命了！雖然這只是一個起點，但你已經擁有了構建智慧、響應式 ML 系統的基石。繼續保持好奇心，勇敢探索 MLOps 的廣闊世界吧！你做得非常棒！