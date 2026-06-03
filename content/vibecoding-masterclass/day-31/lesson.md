哈囉，未來的 MLOps 大師們！

歡迎來到【第 31 天】的挑戰！今天我們要進入一個非常刺激且實用的主題：建立一個自動化的 MLOps 模型更新流程。你可能會想：「哇，MLOps 聽起來好專業、好複雜！」別擔心，我們會用最輕鬆、最白話的方式，一步一步帶你親手實作一個簡易的自動化流程，讓你感受 MLOps 的魅力！

### 【第 31 天：實戰：建立自動化 MLOps 模型更新流程】

#### 為什麼我們需要自動化模型更新？

還記得我們之前訓練好的模型嗎？它們就像小植物，需要陽光、空氣、水才能健康成長。然而，真實世界的資料是會不斷變化的（這就是所謂的「資料漂移 Data Drift」）。你的模型今天表現很好，不代表下週或下個月依然如此。

這時候，MLOps (Machine Learning Operations) 就派上用場了！它就像一個專業的園丁，負責持續監控模型的健康狀況，並在需要時自動進行「修剪」或「更換」：
1.  **定期檢查：** 模型有沒有變笨？
2.  **重新訓練：** 如果有新的更好的數據，重新訓練一個更聰明的模型。
3.  **自動部署：** 如果新模型表現更好，就讓它上線服務，替換掉舊模型。

今天，我們的目標就是建立這個「檢查 -> 訓練 -> 部署」的簡化自動化流程！

#### 我們將會怎麼做？

我們將建立一個 Python 腳本，模擬以下步驟：
1.  **初始模型部署：** 首次訓練一個模型並儲存起來。
2.  **模擬自動更新檢查：** 運行一個函數，它會：
    *   載入目前部署的模型。
    *   模擬有「新資料」進來，用新資料訓練一個「潛在的新模型」。
    *   比較新舊模型的表現。
    *   如果新模型表現更好，就用它替換掉舊模型。
3.  **使用最新模型：** 隨時載入當前最好的模型進行預測。

是不是很有趣？讓我們動手吧！

#### 準備工作

確保你安裝了必要的函式庫：
```bash
pip install scikit-learn pandas
```

#### 程式碼實戰：你的第一個 MLOps 自動化流程

```python
import pandas as pd
from sklearn.datasets import load_iris # 使用鳶尾花數據集
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle # 用於保存和載入模型
import os # 用於檔案系統操作
import random # 用於模擬隨機性

# 定義模型儲存的路徑
MODEL_PATH = 'best_model.pkl'
# 定義一個用於記錄當前最佳準確率的檔案（簡化處理，實際可能用資料庫）
ACCURACY_RECORD_PATH = 'best_accuracy.txt'

# --- 1. 初始模型訓練與儲存 (首次部署) ---
def initial_deploy():
    print("--- 1. 執行初始模型訓練與部署 ---")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    initial_model = LogisticRegression(max_iter=1000)
    initial_model.fit(X_train, y_train)
    initial_accuracy = accuracy_score(y_test, initial_model.predict(X_test))
    print(f"初始模型訓練完成，準確率: {initial_accuracy:.4f}")

    # 儲存初始模型
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(initial_model, f)
    # 儲存初始模型的準確率
    with open(ACCURACY_RECORD_PATH, 'w') as f:
        f.write(str(initial_accuracy))

    print(f"初始模型 '{MODEL_PATH}' 和準確率 '{ACCURACY_RECORD_PATH}' 已儲存。\n")
    return initial_accuracy

# --- 2. 模擬模型自動更新流程 ---
def check_and_update_model():
    print("--- 2. 啟動模型自動更新檢查流程 ---")

    current_best_accuracy = 0.0
    # 載入當前最佳準確率
    if os.path.exists(ACCURACY_RECORD_PATH):
        with open(ACCURACY_RECORD_PATH, 'r') as f:
            current_best_accuracy = float(f.read())
        print(f"當前部署模型的最佳準確率: {current_best_accuracy:.4f}")
    else:
        print("沒有找到現有的模型準確率紀錄，請先運行初始部署。")
        return

    # 模擬新資料 (這裡我們為了演示效果，每次都用不同的 random_state 重新分割數據)
    # 實際應用中，這會是你從資料庫、資料湖或API取得的真實新資料
    print("模擬取得新資料並重新訓練模型...")
    iris = load_iris()
    X_new, y_new = iris.data, iris.target

    # 使用動態的 random_state 模擬數據分佈的變化
    random_state_for_retraining = random.randint(0, 100)
    X_train_retrain, X_test_retrain, y_train_retrain, y_test_retrain = train_test_split(
        X_new, y_new, test_size=0.2, random_state=random_state_for_retraining
    )

    new_model = LogisticRegression(max_iter=1000)
    new_model.fit(X_train_retrain, y_train_retrain)
    new_accuracy = accuracy_score(y_test_retrain, new_model.predict(X_test_retrain))
    print(f"新訓練模型的準確率: {new_accuracy:.4f}")

    if new_accuracy > current_best_accuracy:
        print("🎉 新模型表現更好！正在更新模型...")
        # 儲存新模型
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(new_model, f)
        # 更新最佳準確率紀錄
        with open(ACCURACY_RECORD_PATH, 'w') as f:
            f.write(str(new_accuracy))
        print("模型和準確率已成功更新！")
    else:
        print("新模型表現不如或與當前部署的模型相同，無需更新。")

    print("--- 模型自動更新檢查流程結束 ---\n")

# --- 3. 載入並使用最新模型進行預測 ---
def predict_with_latest_model(sample_data):
    print("--- 3. 載入並使用最新模型進行預測 ---")
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            latest_model = pickle.load(f)
        
        prediction = latest_model.predict(sample_data)
        print(f"使用當前最佳模型對 {sample_data} 進行預測：類別 {prediction[0]}")
        # 鳶尾花數據集類別：0:setosa, 1:versicolor, 2:virginica
        
        # 也可以再次評估其在一個固定測試集上的性能，確保模型依然有效
        iris = load_iris()
        _, X_test, _, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        current_deployed_accuracy = accuracy_score(y_test, latest_model.predict(X_test))
        print(f"目前部署模型的準確率 (在固定測試集上): {current_deployed_accuracy:.4f}")

    else:
        print("沒有可用的模型進行預測，請先訓練並部署模型。")

# 主要執行區塊
if __name__ == "__main__":
    # 模擬首次部署
    print("==== 模擬首次部署模型 ====")
    initial_deploy()

    # 模擬第一次自動更新檢查 (可能會更新模型)
    print("==== 模擬模型自動更新檢查 (第一次) ====")
    check_and_update_model()

    # 模擬第二次自動更新檢查 (可能因為隨機性而不會更新，或再次更新)
    print("==== 模擬模型自動更新檢查 (第二次) ====")
    check_and_update_model()
    
    # 模擬第三次自動更新檢查 (可能因為隨機性而不會更新，或再次更新)
    print("==== 模擬模型自動更新檢查 (第三次) ====")
    check_and_update_model()

    # 載入並使用最新模型進行預測
    print("==== 載入並使用最新模型進行預測 ====")
    # 鳶尾花數據的範例，例如：[萼片長度, 萼片寬度, 花瓣長度, 花瓣寬度]
    sample_data_point = [[5.1, 3.5, 1.4, 0.2]] 
    predict_with_latest_model(sample_data_point)

    print("\n恭喜你！你已經建立了一個簡化的自動化 MLOps 模型更新流程！")

```

#### 程式碼解釋與重點

1.  **`initial_deploy()` 函數：**
    *   這是你的模型第一次「上線」。我們訓練一個 Logistic Regression 模型，計算其準確率，然後用 `pickle` 將模型物件序列化（保存）到 `best_model.pkl` 檔案中。
    *   我們也將模型的準確率儲存在 `best_accuracy.txt`，這是為了方便後續比較。

2.  **`check_and_update_model()` 函數：**
    *   這是 MLOps 自動化的核心！它會讀取 `best_accuracy.txt` 來獲取當前部署模型的最佳準確率。
    *   **模擬新資料：** 我們用 `random.randint(0, 100)` 來改變 `train_test_split` 的 `random_state`，這會導致每次訓練時數據切分略有不同，從而模擬出模型在「新資料」上的表現差異。
    *   **重新訓練與評估：** 使用這些「新資料」訓練一個新的模型，並評估其準確率。
    *   **條件式更新：** 比較新模型的準確率是否優於 `current_best_accuracy`。
        *   如果更好：太棒了！我們將新模型保存到 `best_model.pkl` (覆蓋舊模型)，並更新 `best_accuracy.txt`。這就是模型的「自動部署」！
        *   如果沒有更好：舊模型繼續服務，無需更新。

3.  **`predict_with_latest_model()` 函數：**
    *   無論模型有沒有更新，我們的預測服務都只會載入 `best_model.pkl`。這確保你總是在使用當前表現最好的模型來提供服務。

4.  **`if __name__ == "__main__":` 區塊：**
    *   這個區塊定義了程式的執行順序。
    *   我們首先調用 `initial_deploy()` 進行首次模型部署。
    *   然後，我們多次調用 `check_and_update_model()`，模擬模型在不同時間點進行自動檢查和更新。你可以觀察每次檢查後，模型是否被更新。
    *   最後，我們使用 `predict_with_latest_model()` 來測試最新的模型。

#### 如何運行？

1.  將上述程式碼儲存為 `mlops_pipeline.py`。
2.  打開你的終端機或命令提示字元。
3.  執行命令：`python mlops_pipeline.py`

你會看到一系列的輸出，顯示模型是如何被訓練、檢查、以及是否更新。每次運行 `check_and_update_model()` 時，由於 `random_state` 的變化，新模型的表現可能會不同，從而觸發或不觸發更新。

#### 恭喜你！

你已經成功建立了一個簡化的 MLOps 自動化模型更新流程！這是一個非常重要的概念，讓你從單純的模型開發者，晉升為能夠管理模型生命週期的工程師。

**下一步可以怎麼做？**

*   **真正的資料流：** 思考如何從真實的資料庫或 API 獲取新資料，而不是模擬。
*   **模型版本管理：** 不要直接覆蓋舊模型，而是保存不同版本的模型 (例如 `model_v1.pkl`, `model_v2.pkl`)，並記錄每個版本的性能。
*   **監控工具：** 學習使用像 MLflow 這樣的工具來追蹤模型的實驗、版本和部署。
*   **CI/CD：** 探索 Jenkins, GitHub Actions, GitLab CI/CD 等工具，讓你的 `check_and_update_model` 腳本真的能夠在伺服器上定期自動執行。
*   **模型回滾：** 如果新模型表現不如預期，如何快速切換回舊模型？

今天的實戰讓你對 MLOps 有了初步的認識。記住，這只是冰山一角，但你已經邁出了重要的一步！持續學習，你將會在機器學習領域走得更遠！

我們明天見！😊