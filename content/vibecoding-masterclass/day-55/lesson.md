哈囉，我的程式學習夥伴！恭喜你，我們又來到了一個重要的里程碑：**第 55 天**！

在這趟學習旅程中，我們已經學會了如何收集資料、處理資料、訓練出厲害的模型。但你曾想過嗎？這些在你的電腦上跑得好好的模型，要怎麼讓真實世界的人們也能使用到它們的魔力呢？當你的模型越來越多、越來越好時，又要怎麼管理它們，確保大家永遠都在用「最新、最好」的版本呢？

別擔心！今天，我們要揭開一個超級酷炫的主題：**MLOps 自動化部署與模型版本管理**。

### 【第 55 天：實戰：MLOps 自動化部署與模型版本管理】

**MLOps (Machine Learning Operations)** 簡單來說，就是把軟體開發領域的 DevOps 精神，應用到機器學習專案上。它幫助我們更有效率、更可靠地**部署 (Deploy)**、**監控 (Monitor)** 和**更新 (Update)** 機器學習模型。

聽起來有點嚴肅？其實它會讓你的生活更輕鬆，讓模型不再只是你電腦裡的「寶貝」，而是能真正服務大眾的「利器」！

---

### 第一步：模型部署小試身手 (Deployment – A First Try!)

想像一下，你訓練好了一個預測房價的模型，你想讓網頁應用程式呼叫它來提供預測服務。這時候，我們需要將模型「部署」成一個 API 服務。

這裡我們用 Python 的輕量級網頁框架 **Flask** 來做個簡單的範例。

**1. 準備你的模型**

首先，我們需要一個已經訓練好的模型。我們通常會把它序列化 (serialize) 存檔，以便之後載入。這裡我們用 `joblib` (或者 `pickle`)。

```python
# model.py
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# 假設你已經有了這些訓練數據
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# 訓練一個簡單的線性迴歸模型
model = LinearRegression()
model.fit(X, y)

# 將模型存檔，命名為 'model_v1.pkl' (代表這是第一版)
joblib.dump(model, 'model_v1.pkl')
print("模型 v1 已經成功存檔：model_v1.pkl")
```

先執行 `python model.py`，你就會得到一個 `model_v1.pkl` 檔案。

**2. 建立 Flask API 服務**

接下來，我們寫一個 Flask 應用程式，它會載入這個模型，並提供一個 `/predict` 的 API 接口。當有人發送數據過來，它就用模型進行預測並回傳結果。

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# 定義模型路徑
MODEL_PATH = os.getenv('MODEL_PATH', 'model_v1.pkl') # 可以透過環境變數設定

# 載入模型
try:
    model = joblib.load(MODEL_PATH)
    print(f"成功載入模型：{MODEL_PATH}")
except Exception as e:
    print(f"載入模型失敗：{e}")
    model = None # 載入失敗則模型為None，避免後續報錯

@app.route('/')
def home():
    return "歡迎來到我們的 ML 預測服務！請透過 /predict 接口發送數據。"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '模型尚未載入，請聯繫管理員。'}), 500

    try:
        data = request.get_json(force=True) # 接收 JSON 格式的輸入
        features = np.array(data['features']).reshape(1, -1) # 轉換成模型需要的格式
        prediction = model.predict(features)[0] # 進行預測
        return jsonify({'prediction': float(prediction)}) # 回傳預測結果
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # 在本機運行，debug=True 會在程式碼變動時自動重啟
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**3. 運行並測試**

1.  確保你安裝了必要的套件：`pip install flask scikit-learn joblib numpy`
2.  執行 `python app.py`。你會看到類似 `* Running on http://0.0.0.0:5000/` 的訊息。
3.  打開你的終端機，用 `curl` 來測試看看：

    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{"features": [10]}' \
         http://127.0.0.1:5000/predict
    ```

    你應該會得到類似 `{"prediction": 10.9}` 的結果。是不是很有趣？你的模型現在可以透過網路被訪問了！

---

### 第二步：模型版本管理 (Model Version Management)

現實世界中，模型不是一成不變的。我們可能會收集到更多數據，或者找到更好的演算法，訓練出一個效能更好的模型。這時候，我們就需要**模型版本管理**。

想像你訓練出了一個更好的模型：

```python
# model_v2.py (或直接更新 model.py 內容)
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# 假設你用更多數據或調整了參數，得到一個「更好」的模型
X_new = np.array([[1], [2], [3], [4], [5], [6], [7]])
y_new = np.array([2, 4, 5, 4, 5, 7, 8]) # 稍微不同的數據

model_v2 = LinearRegression()
model_v2.fit(X_new, y_new)

# 將新的模型存檔，命名為 'model_v2.pkl'
joblib.dump(model_v2, 'model_v2.pkl')
print("模型 v2 已經成功存檔：model_v2.pkl")
```

執行 `python model_v2.py` 後，你現在有了 `model_v1.pkl` 和 `model_v2.pkl`。

要讓你的 Flask 服務使用 `model_v2.pkl`，最簡單的方式就是修改 `app.py` 中的 `MODEL_PATH` 變數：

```python
# app.py (更新版本)
# ...
MODEL_PATH = os.getenv('MODEL_PATH', 'model_v2.pkl') # 將預設模型改為 v2
# ...
```

然後重啟 `python app.py`。現在你的服務就會載入並使用 `model_v2` 進行預測了！

**更進一步的思考：**
在真實世界的 MLOps 中，模型版本管理通常會更複雜，例如：

*   **專門的版本管理工具：** 如 MLflow、DVC (Data Version Control) 等，可以追蹤模型的訓練參數、指標、來源數據等。
*   **動態載入：** 服務可能根據配置或環境變數，動態決定要載入哪個版本的模型，甚至可以 AB 測試不同模型。

---

### 第三步：自動化部署的想像 (Imagining Automated Deployment)

現在你看到了，當模型更新時，我們需要手動修改程式碼，然後重啟服務。這在開發初期還好，但在大型專案中，這絕對是個夢魘！這就是**自動化部署 (Automated Deployment)** 的用武之地。

自動化部署的核心思想是：當你的程式碼或模型發生變化（例如，你把新的模型程式碼 push 到 GitHub），一個預設好的流程就會自動觸發：

1.  **測試 (Testing):** 確保新程式碼沒有錯誤。
2.  **建構 (Building):** 可能會打包你的應用程式（例如，製作一個 Docker 映像）。
3.  **部署 (Deployment):** 將新的應用程式（包含新的模型）自動部署到伺服器上，替換掉舊的版本。

這整個流程通常被稱為 **CI/CD (Continuous Integration/Continuous Deployment)**。

我們這裡用一個概念性的 **GitHub Actions** 範例來想像一下：

```yaml
# .github/workflows/deploy.yml (概念性範例，實際部署會更複雜)
name: MLOps Model Deploy Pipeline

on:
  push:
    branches:
      - main # 當有新的程式碼推送到 main 分支時觸發此工作流程

jobs:
  deploy:
    runs-on: ubuntu-latest # 在 Ubuntu 虛擬機上運行
    steps:
    - name: Checkout code # 檢查程式碼
      uses: actions/checkout@v2

    - name: Set up Python # 設定 Python 環境
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Install dependencies # 安裝必要的套件
      run: |
        pip install flask scikit-learn joblib numpy

    - name: Train and save model (for demonstration) # 這裡可以觸發模型訓練
      run: |
        python model_v2.py # 假設這個腳本會訓練並保存最新的模型

    - name: Deploy model and service # 實際部署步驟 (這裡只是模擬)
      run: |
        # 在真實世界中，這裡會包含：
        # 1. 將 Flask 應用打包成 Docker 映像
        # 2. 將 Docker 映像推送到容器倉庫 (如 Docker Hub, AWS ECR)
        # 3. 更新雲端服務 (如 Kubernetes, AWS Lambda, Google Cloud Run) 的部署設定
        #    讓它們拉取最新的 Docker 映像並啟動服務
        echo "模擬：將 model_v2 和 Flask 服務自動部署至伺服器..."
        echo "部署完成！您的服務現在使用最新的模型版本！"
        # 為了讓app.py使用新模型，你可能需要重啟遠端服務，或透過環境變數設定
```

當你把這個 YAML 檔案放在專案的 `.github/workflows/` 目錄下，並將程式碼推送到 `main` 分支時，GitHub 就會自動執行這些步驟！這就實現了**自動化**！

---

### 結語

恭喜你！今天我們探索了 MLOps 世界的冰山一角：模型部署、版本管理以及自動化部署的基本概念。從把你的模型變成一個網路服務，到優雅地管理不同版本的模型，再到想像自動化部署的流程，這都是你將 ML 知識轉化為實際應用服務的關鍵步驟。

這只是個開始！MLOps 是一個廣闊的領域，還有很多酷炫的工具和技術（如 Docker、Kubernetes、MLflow、Airflow 等待你去探索。但別擔心，今天你已經邁出了非常重要的一步！

繼續保持這份好奇心和熱情，期待你在 MLOps 的旅程中創造更多奇蹟！我們下一個主題見！