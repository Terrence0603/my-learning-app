哈囉！各位熱情的程式學習者們！ 🚀

恭喜你們走到第 198 天！這段旅程充滿了探索與學習，而今天我們要來聊一個讓你的機器學習專案更上一層樓的關鍵話題：**MLOps 中的模型再訓練與版本控制**。

別被「MLOps」這個聽起來有點專業的詞嚇到！簡單來說，MLOps 就是把「機器學習 (ML)」和「維運 (Ops)」結合起來，讓你的模型能夠更穩定、更可靠地在真實世界中運作，並且能持續進步。而今天我們要專注的兩大重點，就是讓你的模型「**保持新鮮**」(再訓練) 和「**不迷失方向**」(版本控制)。

### 為什麼模型需要「再訓練」？🤔

想像一下，你辛辛苦苦訓練了一個模型，它在你提供的數據上表現得很好。但隨著時間推移，世界在變！新的數據出現了，使用者行為改變了，甚至你的產品需求也有可能調整。這時候，你原本的模型可能會變得「過時」，預測準確度下降。

這就像你學了一項技能，剛開始很厲害，但如果不再練習，很快就會被新的技術超越。所以，**模型再訓練**就是定期用最新的數據來「喚醒」你的模型，讓它重新學習，保持最佳狀態。

### 版本控制：你的時間機器！🕰️

那「版本控制」又是什麼呢？想像一下，你修改了一份重要的文件，改了好幾個版本，結果不小心把一個比較舊但正確的版本覆蓋掉了！是不是很崩潰？

在模型開發中，情況更複雜。你可能會嘗試不同的超參數、不同的模型架構、不同的特徵工程方法。如果沒有好好記錄，你可能不知道哪個版本在什麼情況下表現最好，也無法輕鬆回溯到之前的某個版本。

**版本控制**就像你的「時間機器」，它會記錄你每一次的實驗、每一次模型的修改，讓你能夠追溯、比較，甚至一鍵回到過去！這對於排查問題、實驗對比，以及確保專案的穩定性至關重要。

### 來點實際的！動手寫程式碼吧！ 💻

說了這麼多，我們來看看如何在 Python 中實現這些概念。這裡我們將使用 `mlflow` 這個超讚的開源工具，它能幫我們輕鬆管理模型訓練的實驗和版本。

首先，確保你已經安裝了 `mlflow`：

```bash
pip install mlflow scikit-learn pandas
```

接下來，我們來寫一段模擬模型訓練和記錄版本控制的程式碼：

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 模擬一些訓練數據
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100) * 2,
    'target': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

X = df[['feature1', 'feature2']]
y = df['target']

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 模型訓練與版本記錄 ---

# 為了示範，我們模擬兩次不同的訓練
for i in range(2):
    # 設置一個實驗名稱
    experiment_name = "model_retraining_demo"
    mlflow.set_experiment(experiment_name)

    # 啟動一個 MLflow 運行 (Run)
    # 每個運行都代表一次獨立的實驗
    with mlflow.start_run(run_name=f"Training Run {i+1}"):
        print(f"--- Starting Training Run {i+1} ---")

        # 這裡可以根據需要調整模型參數，模擬再訓練或微調
        if i == 0:
            model_params = {"C": 1.0, "solver": "liblinear", "random_state": 42}
            print("Using initial model parameters.")
        else:
            # 模擬第二次訓練，可能調整了參數或使用了不同的數據處理
            model_params = {"C": 1.5, "solver": "liblinear", "random_state": 42}
            print("Using adjusted model parameters for retraining.")

        # 創建並訓練模型
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        # 進行預測
        y_pred = model.predict(X_test)

        # 計算準確度
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for Run {i+1}: {accuracy:.4f}")

        # --- 記錄到 MLflow ---

        # 記錄模型參數
        mlflow.log_params(model_params)

        # 記錄指標 (Metrics)
        mlflow.log_metric("accuracy", accuracy)

        # 記錄模型本身 (這就完成了模型的版本控制!)
        # MLflow 會為你儲存這個模型的特定版本
        mlflow.sklearn.log_model(model, "model")

        print(f"--- Finished Training Run {i+1} ---")

print("\nMLflow runs have been logged. You can view them by running 'mlflow ui' in your terminal.")
```

**程式碼說明：**

1.  我們模擬了一些數據，並將其分割為訓練集和測試集。
2.  我們進入一個迴圈，模擬了兩次不同的模型訓練。
3.  `mlflow.set_experiment("model_retraining_demo")`：設定我們這次實驗的名稱，方便管理。
4.  `with mlflow.start_run(...)`：每一次的訓練（或調整）都會被包在一個 `run` 裡。MLflow 會自動為每個 `run` 分配一個唯一的 ID。
5.  `mlflow.log_params(...)`：記錄我們這次訓練使用的模型參數。
6.  `mlflow.log_metric("accuracy", accuracy)`：記錄我們關心的模型指標，例如準確度。
7.  `mlflow.sklearn.log_model(model, "model")`：**這是關鍵！** 它會將訓練好的模型保存起來，並在 MLflow 中註冊為一個新的「模型版本」。下次你想用這個模型時，就可以透過 MLflow 找到它。

### 如何查看你的「時間機器」？

在你執行完上面的 Python 程式碼後，請打開你的終端機 (Terminal)，並切換到你執行程式碼的目錄，然後輸入：

```bash
mlflow ui
```

接著，在你的瀏覽器中打開 `http://localhost:5000`。你就能看到一個網頁介面，裡面詳細記錄了你剛剛進行的兩次訓練，包含了參數、指標，以及最重要的，每個模型的可下載版本！是不是很神奇？✨

### 總結

模型再訓練讓我們能保持模型的「新鮮度」，而版本控制則讓我們能追溯、比較，並確保我們能隨時回到最佳狀態。MLOps 的工具，如 MLflow，正是為了解決這些實際問題而生的。

今天我們只是觸及了 MLOps 的皮毛，但希望這個小小的練習能讓你對模型再訓練和版本控制有更實際的理解。隨著你的專案越來越龐大，這些能力將會是你穩定、高效開發的基石！

繼續保持你的學習熱情，我們下一次見！👋