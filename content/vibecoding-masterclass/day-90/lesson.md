哈囉，親愛的程式學習夥伴們！歡迎來到我們的「第 90 天」旅程！

是不是感覺時間過得飛快？從一開始的變數、迴圈，到現在我們已經能討論機器學習模型，甚至要來建構自動化的 MLOps 管線了！這趟旅程真是令人興奮，而今天，我們要將之前學到的知識融會貫通，一起打造一個超級酷炫的 **端到端 MLOps 持續整合與交付 (CI/CD) 管線**！

別緊張，聽起來很專業，但我們會用最輕鬆、最鼓勵的方式，帶你一步步理解並實作。想像一下，未來你只需要將訓練好的模型程式碼推送到 Git 倉庫，後面的一切（測試、打包、部署）都自動完成了，是不是很省心？這就是 MLOps CI/CD 的魔力！

---

## 【第 90 天：實戰：建構端到端 MLOps 持續整合與交付管線】

### 一、 MLOps CI/CD，究竟在忙些什麼？

MLOps 簡單來說，就是把軟體工程 (DevOps) 的最佳實踐應用到機器學習專案上。而 CI/CD 則是其中的核心：

*   **CI (Continuous Integration, 持續整合)**：當你提交程式碼時，自動檢查程式碼品質、執行單元測試、確保新舊程式碼能和諧運作。
*   **CD (Continuous Delivery/Deployment, 持續交付/部署)**：CI 階段通過後，自動將你的模型打包成可運行的環境（例如 Docker 容器），並部署到伺服器上，隨時提供服務。

這樣一來，我們就能更快、更穩定地將模型從實驗室推向實際應用，減少人為錯誤，提升效率！

### 二、我們的「玩具專案」目標

為了讓大家快速上手，我們今天會建構一個極度簡化的 MLOps 管線：

1.  有一個 Python 腳本，假裝「訓練」了一個模型並保存。
2.  使用 Docker 將這個「模型訓練」環境打包。
3.  利用 GitHub Actions 作為 CI/CD 工具，當程式碼提交時：
    *   執行程式碼品質檢查。
    *   建構 Docker 映像檔。
    *   「模擬」模型部署。

準備好了嗎？我們開始吧！

### 三、動手實作：程式碼與配置

首先，請在你的專案資料夾中建立以下三個檔案：

1.  `train.py`：我們的「模型訓練」腳本。
2.  `requirements.txt`：Python 依賴列表。
3.  `Dockerfile`：打包模型的 Dockerfile。
4.  `.github/workflows/main.yml`：GitHub Actions 的 CI/CD 配置。

#### 1. `train.py` (我們的假模型訓練腳本)

```python
# train.py
import pickle
import os

def train_model():
    """模擬一個機器學習模型訓練的過程，並保存一個假模型檔案。"""
    print("🚀 開始訓練模型！這是一個簡化的模擬過程...")

    # 這裡我們不真的訓練模型，只是創建一個假的模型資料
    # 在實際情況中，這裡會是你的 scikit-learn, TensorFlow 或 PyTorch 模型訓練代碼
    model_data = {
        "version": "1.0",
        "algorithm": "DummyRegressor",
        "parameters": {"n_estimators": 100},
        "accuracy": 0.92,
        "message": "這是一個模擬的模型，訓練於 Day 90 MLOps 課程！"
    }

    # 將模型資料保存為 pickle 檔案
    model_filename = "model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)

    print(f"🎉 模型訓練完成，並已保存為 '{model_filename}'")
    print("你可以想像這個模型現在已經準備好提供預測服務了！")

if __name__ == "__main__":
    train_model()
```

#### 2. `requirements.txt` (Python 依賴)

```
# requirements.txt
# 在這個簡單的例子中，我們只使用了標準庫的 pickle
# 但在真實的 ML 專案中，這裡會包含 scikit-learn, pandas, numpy 等
```
*小提示：即使是空的，也習慣性放一個，方便 Dockerfile 使用。*

#### 3. `Dockerfile` (模型環境打包)

```dockerfile
# Dockerfile
# 使用一個輕量級的 Python 映像檔作為基礎
FROM python:3.9-slim-buster

# 設定工作目錄，所有操作都在這個目錄下進行
WORKDIR /app

# 將 requirements.txt 複製到容器中，並安裝依賴
# --no-cache-dir 可以減少映像檔大小
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 將我們的模型訓練腳本複製到容器中
COPY train.py .

# 定義容器啟動時執行的命令
# 在這裡，我們讓它執行模型訓練腳本
CMD ["python", "train.py"]
```
*是不是感覺很熟悉？我們之前學過 Docker 了！*

#### 4. `.github/workflows/main.yml` (GitHub Actions CI/CD 配置)

在你的專案根目錄下，建立 `.github` 資料夾，然後在裡面再建立 `workflows` 資料夾，最後在 `workflows` 裡建立 `main.yml`。

```yaml
# .github/workflows/main.yml
name: MLOps CI/CD Pipeline (Day 90)

on:
  push:
    branches:
      - main # 當有程式碼推送到 main 分支時，觸發此管線

jobs:
  build_and_deploy_ml_model:
    runs-on: ubuntu-latest # 在 Ubuntu 系統上執行這個 Job

    steps:
      - name: 🚀 檢查程式碼 (Checkout repository)
        uses: actions/checkout@v3 # 使用 GitHub 提供的 action 來檢查倉庫程式碼

      - name: 🐍 設定 Python 環境
        uses: actions/setup-python@v4
        with:
          python-version: '3.9' # 指定使用的 Python 版本

      - name: ✨ 安裝 Python 依賴 (用於品質檢查或測試)
        run: |
          python -m pip install --upgrade pip
          # pip install -r requirements.txt # 如果有其他依賴需要用於測試，可以在這裡安裝
          # pip install flake8 # 為了程式碼品質檢查，我們安裝 flake8
          echo "Python 環境準備就緒！"

      - name: 🔍 執行程式碼品質檢查 (Linting)
        run: |
          echo "執行程式碼品質檢查中... (例如使用 flake8)"
          # 在真實專案中，這裡會運行如 `flake8 .` 這樣的命令
          echo "程式碼品質檢查通過！代碼棒棒噠！"

      - name: 🧪 執行模型相關測試 (單元測試/整合測試)
        run: |
          echo "執行模型相關測試中... (例如 pytest, 模型性能測試)"
          # 在真實專案中，這裡會運行如 `pytest` 這樣的命令
          echo "所有測試都通過了！模型很穩定！"

      - name: 🐳 建構 Docker 映像檔
        run: |
          docker build -t my-ml-model:latest .
          echo "Docker 映像檔建構完成：my-ml-model:latest"
          echo "這代表我們的模型及其運行環境已經被完美打包了！"

      - name: 🚢 模擬模型部署到生產環境
        run: |
          echo "================================================="
          echo "🎉🎉🎉 CI/CD 管線執行到部署階段啦！ 🎉🎉🎉"
          echo "想像一下，現在這個 'my-ml-model:latest' 映像檔"
          echo "正在被推送到 Docker Hub 或私有容器註冊表，"
          echo "然後自動部署到 Kubernetes 叢集、AWS SageMaker 或 Google Cloud AI Platform！"
          echo "你的模型已經上線，開始為用戶提供預測服務了！"
          echo "================================================="
```

### 四、親自體驗 MLOps CI/CD 的魔力！

1.  **初始化 Git 倉庫並推送到 GitHub：**
    在你的專案根目錄執行：
    ```bash
    git init
    git add .
    git commit -m "Day 90 MLOps pipeline initial setup"
    git branch -M main
    git remote add origin YOUR_GITHUB_REPO_URL # 替換成你的 GitHub 倉庫 URL
    git push -u origin main
    ```

2.  **觀察 GitHub Actions 執行：**
    推送到 GitHub 後，打開你的 GitHub 倉庫頁面，點擊上方的 "Actions" 選項卡。你應該會看到一個正在運行的工作流程，名稱就是 "MLOps CI/CD Pipeline (Day 90)"。點擊進去，你可以看到每個步驟都在自動執行！

    *   `檢查程式碼`
    *   `設定 Python 環境`
    *   `安裝 Python 依賴`
    *   `執行程式碼品質檢查`
    *   `執行模型相關測試`
    *   `建構 Docker 映像檔`
    *   `模擬模型部署到生產環境`

    當所有步驟都綠色打勾，表示你的第一個 MLOps CI/CD 管線就成功跑通了！是不是超有成就感！

### 五、結語與展望

恭喜你，夥伴！你成功在第 90 天建構了一個簡化的端到端 MLOps CI/CD 管線！這是一個非常重要的里程碑。你學會了如何將程式碼版本控制、容器化技術以及自動化流程結合起來，讓機器學習模型的開發與部署變得更加高效和可靠。

當然，這只是一個開始。真實世界的 MLOps 管線會更複雜，包含更多階段，例如：

*   **數據版本控制 (DVC)**：追蹤數據集變化。
*   **模型註冊表 (Model Registry)**：管理不同版本的模型。
*   **A/B 測試**：部署新舊模型進行效果比較。
*   **模型監控 (Model Monitoring)**：追蹤模型在生產環境中的性能。
*   **更複雜的部署策略**：如藍綠部署、金絲雀部署。

但今天的實作，已經為你打下了堅實的基礎。從現在開始，你已經具備了將機器學習模型從「程式碼」變為「服務」的思維和初步技能。

繼續保持這樣的好奇心和動手能力，你會在 MLOps 的世界裡走得更遠！為你的第 90 天喝采！你真的太棒了！