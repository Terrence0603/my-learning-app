好的，各位未來的 MLOps 大師們！歡迎來到【第 41 天】的旅程！

---

## 【第 41 天：實戰：MLOps 模型容器化與生產部署】

嗨，各位程式探險家！恭喜你，你已經走到第 41 天了！這意味著你已經學習了大量關於機器學習的知識，從資料處理到模型訓練，你都做得非常棒！

但你有沒有想過，當你的模型在你的電腦上表現得天衣無縫時，要怎麼讓全世界都能用它呢？或者，當你的同事想用你的模型時，卻發現環境設定、依賴套件的版本問題一大堆，怎麼辦？

這就是我們今天的主題——**MLOps (Machine Learning Operations)** 的核心精神：**讓你的模型能夠穩定、可靠、重複地從實驗室走向真實世界！** 而今天，我們將專注於兩個關鍵步驟：**模型容器化** 和 **生產部署**。聽起來有點高大上，但相信我，我們會用最輕鬆的方式帶你一步步完成！

### 為什麼需要容器化？

你可能聽過這句話：「在我的機器上跑得好好的啊！」(It works on my machine!)。這是工程師最常遇到的魔咒。你的模型可能需要特定版本的 Python、特定的函式庫（例如 `scikit-learn` 或 `tensorflow`），以及它們的特定版本。當你把模型程式碼給別人時，他們可能需要花費大量時間來複製你的環境。

**容器化 (Containerization)**，特別是使用 **Docker**，就是來解決這個問題的。想像一下，Docker 就像一個魔法盒子，你可以把你的模型、所有需要的程式碼、Python 環境、函式庫，甚至連作業系統的必要組件，都打包進這個盒子裡。這個盒子可以在任何支援 Docker 的地方運行，而且行為完全一致，無論是你的筆電、公司的伺服器，還是雲端平台！

### 準備工作：一個簡單的模型

為了今天的實作，我們需要一個簡單的模型。我們將使用 `scikit-learn` 來訓練一個非常基本的模型，並將它保存起來。

首先，在你的專案資料夾裡，建立一個 `train_model.py` 檔案：

```python
# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# 載入 Iris 資料集
iris = load_iris()
X, y = iris.data, iris.target

# 訓練一個簡單的羅吉斯迴歸模型
model = LogisticRegression(max_iter=200) # 增加迭代次數避免收斂警告
model.fit(X, y)

# 將訓練好的模型保存為 pickle 檔案
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("模型已成功訓練並保存為 model.pkl")
```

然後，在終端機中執行它：`python train_model.py`。這會生成一個 `model.pkl` 檔案，這就是我們要部署的模型。

### 第一步：為模型建立 API 服務 (app.py)

模型要讓大家使用，最常見的方式就是透過 **API (Application Programming Interface)**。當別人發送資料給你的 API，你的模型就會返回預測結果。我們使用輕量級的 **Flask** 框架來建立這個 API。

在你的專案資料夾中，建立一個 `app.py` 檔案：

```python
# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# 載入之前保存的模型
# 確保 model.pkl 在相同目錄下
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "歡迎使用我的 ML 模型 API！請透過 /predict 路徑發送 POST 請求。"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 從請求中獲取 JSON 資料
        data = request.get_json(force=True)
        # 假設輸入資料是像 [5.1, 3.5, 1.4, 0.2] 這樣的特徵列表
        features = np.array(data['features']).reshape(1, -1)

        # 使用模型進行預測
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features).tolist() # 預測機率

        # 返回預測結果
        return jsonify({
            'prediction': prediction[0].item(), # .item() 將 numpy int 轉換為 Python int
            'probabilities': prediction_proba[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # 讓 Flask 應用程式在所有可用的 IP 位址 (0.0.0.0) 和 5000 埠上運行
    app.run(host='0.0.0.0', port=5000)
```

**建立 `requirements.txt`：**
為了讓 Docker 知道需要安裝哪些套件，請建立一個 `requirements.txt` 檔案：

```
Flask==2.3.2
scikit-learn==1.3.0
numpy==1.26.0
```
*(注意：版本號可能會因你的環境而異，你可以使用 `pip freeze > requirements.txt` 來生成你目前環境的套件列表，然後精簡它。)*

此時你的專案資料夾應該包含：`train_model.py`, `model.pkl`, `app.py`, `requirements.txt`。

### 第二步：容器化模型 (Dockerfile)

現在，重頭戲來了！我們要告訴 Docker 如何建立這個魔法盒子。在你的專案資料夾中，建立一個名為 `Dockerfile` (注意：沒有副檔名) 的檔案：

```dockerfile
# Dockerfile

# 1. 指定基礎影像：我們選擇一個輕量級的 Python 3.9 官方影像
FROM python:3.9-slim-buster

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 將 requirements.txt 複製到容器的工作目錄
# 這一步單獨複製是為了利用 Docker 的快取機制，如果 requirements.txt 不變，則不會重複安裝
COPY requirements.txt .

# 4. 安裝所有必要的 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 5. 將所有其他專案檔案（包括 app.py 和 model.pkl）複製到容器的工作目錄
COPY . .

# 6. 宣告容器會監聽的埠號 (port)。這裡我們的 Flask 應用程式監聽 5000 埠
EXPOSE 5000

# 7. 定義容器啟動時要執行的命令。這會啟動我們的 Flask 應用程式
CMD ["python", "app.py"]
```

### 第三步：建立並運行 Docker 容器

現在我們有了 `Dockerfile`，我們可以建立並運行我們的模型服務了！

1.  **建立 Docker 影像：**
    在你的終端機中，切換到你的專案資料夾，然後執行以下命令：
    ```bash
    docker build -t my-ml-app .
    ```
    *   `docker build`：建立 Docker 影像的命令。
    *   `-t my-ml-app`：給你的影像一個標籤或名稱，這裡我們叫做 `my-ml-app`。
    *   `.`：告訴 Dockerfile 在當前目錄下。

    你會看到一堆輸出，Docker 正在按照 `Dockerfile` 的指示一步步建立影像。

2.  **運行 Docker 容器：**
    影像建立完成後，我們就可以運行它了：
    ```bash
    docker run -p 5000:5000 my-ml-app
    ```
    *   `docker run`：運行 Docker 容器的命令。
    *   `-p 5000:5000`：這是埠號映射。它將主機電腦的 5000 埠映射到容器內的 5000 埠。這樣你就可以透過 `localhost:5000` 訪問容器內的服務。
    *   `my-ml-app`：指定要運行的影像名稱。

    如果一切順利，你會看到 Flask 應用程式啟動的訊息，類似於：`* Running on http://0.0.0.0:5000`。

### 第四步：測試你的容器化模型

你的模型現在應該在 Docker 容器中運行了！我們可以透過 `curl` 命令來測試它：

打開另一個終端機視窗，輸入：

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
     http://localhost:5000/predict
```

你應該會得到類似這樣的 JSON 回應：

```json
{"prediction":0,"probabilities":[0.98..., 0.01..., 0.00...]}
```
這表示你的模型成功接收了輸入，並返回了預測結果！

### 生產部署的下一步

恭喜！你已經成功地將你的機器學習模型容器化，並在本地運行起來了！這一步是 MLOps 中最關鍵的一環。

**真正的生產部署**通常會將這個 Docker 影像推送到 **雲端平台**（例如 AWS ECR/ECS, Google Cloud Run/GKE, Azure Container Instances/AKS）或公司內部的 Kubernetes 集群中。這些平台會負責管理你的容器，確保它持續運行，處理流量，並在需要時自動擴展。

今天的實作雖然在本地運行，但你的 `my-ml-app` 這個 Docker 影像，就是可以直接部署到任何這些生產環境的「黃金標準」！

### 總結與展望

今天，我們學習了 MLOps 的重要概念——**模型容器化**。我們使用 Docker 將一個簡單的機器學習模型、它的 API 服務和所有依賴項打包成一個獨立、可移植的單元。你現在已經擁有了一個可以在任何地方穩定運行的模型服務！

這是一個巨大的里程碑！從資料到模型再到可部署的服務，你已經走完了很大一部分的 MLOps 旅程。未來，你可以進一步探索持續整合/持續部署 (CI/CD) 管道、模型監控、A/B 測試等更高級的 MLOps 主題。

你做得非常棒！繼續保持好奇心和學習的熱情，MLOps 的世界正等著你去探索！下一次見！