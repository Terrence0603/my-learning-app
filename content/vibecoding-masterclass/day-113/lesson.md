哈囉，未來的大師們！恭喜你一路走到程式學習的第 113 天！這意味著你已經具備了扎實的基礎，準備好迎接更實用、更具挑戰性的主題了。

今天，我們要探索的是 MLOps 的一個核心環節：**模型監控與可觀測性 (Model Monitoring & Observability)**。想像一下，你辛辛苦苦訓練出一個超棒的 AI 模型，它在測試集上表現完美，然後你把它部署上線了。接下來呢？你會不會希望它能永遠維持最佳狀態？

答案是：在現實世界中，沒有模型是永恆不變的！市場趨勢會變、使用者行為會變、資料分佈也會變。你的模型就像開車上路，你不會希望等到引擎燈亮了才發現問題對吧？今天，我們就要來學習如何為你的 AI 模型打造一個「健康監測系統」，讓它在現實世界中也能保持最佳狀態！

---

### **【第 113 天：實戰：MLOps 模型監控與可觀測性建構——讓你的 AI 模型永保青春活力！】**

#### 1. 為什麼模型需要監控與可觀測性？

模型監控（Model Monitoring）就像是模型的健康檢查，我們要持續觀察它的「生命跡象」。它最主要的目的是：

*   **發現數據漂移 (Data Drift)**：生產環境中的輸入資料分佈，可能與訓練時的資料分佈產生變化。例如，你的顧客年齡層突然普遍變年輕了。
*   **發現概念漂移 (Concept Drift)**：輸入特徵與目標之間的關係發生了變化。例如，原本年齡是購買意願的重要指標，但現在社群影響力變得更重要。
*   **性能下降 (Performance Degradation)**：模型預測的準確度、精確度等指標不如預期。
*   **資源消耗異常**：模型服務佔用的記憶體、CPU 等資源是否過高。

而可觀測性（Observability）則更進一步。當模型表現不如預期時，它能幫助我們找出「為什麼」會這樣。監控告訴你「有問題」，可觀測性幫助你回答「是什麼問題？發生在哪裡？」。

#### 2. 動手實作：從零開始建立簡易監控

對於初學者，我們不需要一開始就跳進複雜的 MLOps 工具（雖然它們超棒！）。我們可以先用 Python 搭配一些數據分析函式庫來開始，理解背後的原理。

**我們的目標：** 模擬一個簡單的分類模型，並監控其「輸入資料」與「輸出預測」的關鍵統計分佈，看看它們在生產環境中是否有「漂移」。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns # 用於更美觀的圖表

# 設定matplotlib顯示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac系統
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows系統
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

print("--- 第 113 天：MLOps 模型監控與可觀測性建構 ---")

# --- Step 1: 模擬初始訓練與部署 ---
print("\n[Step 1] 模擬初始訓練與部署...")

# 1.1 生成模擬訓練資料 (假設預測客戶是否會點擊廣告)
np.random.seed(42)
num_samples_train = 1000
age_train = np.random.normal(loc=35, scale=8, size=num_samples_train)
income_train = np.random.normal(loc=50000, scale=15000, size=num_samples_train)
# 簡單的目標變數生成邏輯：年齡越大，收入越高，點擊機率越大
click_prob_train = 1 / (1 + np.exp(-(0.05 * age_train - 0.00002 * income_train / 10 + np.random.normal(0, 0.5, num_samples_train))))
clicked_train = (click_prob_train > 0.5).astype(int)

df_train = pd.DataFrame({
    'age': age_train,
    'income': income_train,
    'clicked': clicked_train
})

X_train = df_train[['age', 'income']]
y_train = df_train['clicked']

# 1.2 訓練一個簡單的邏輯迴歸模型
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# 1.3 取得訓練時的預測分佈 (機率)
train_predictions_proba = model.predict_proba(X_train)[:, 1]

print("模型已訓練完成，並記錄了訓練資料與預測分佈。")

# --- Step 2: 模擬生產環境中的「數據漂移」 ---
print("\n[Step 2] 模擬生產環境中的數據漂移...")

# 2.1 生成模擬生產資料 (假設經過一段時間，客戶群體發生變化)
num_samples_prod = 500
# 假設「年齡」平均值變年輕了 (數據漂移)
age_prod = np.random.normal(loc=28, scale=7, size=num_samples_prod)
# 假設「收入」的分佈變得更廣了 (數據漂移)
income_prod = np.random.normal(loc=55000, scale=20000, size=num_samples_prod)
# 我們不會有生產環境的真實標籤，所以這裡暫不生成 clicked_prod

df_prod = pd.DataFrame({
    'age': age_prod,
    'income': income_prod
})

X_prod = df_prod[['age', 'income']]

# 2.2 使用部署的模型進行生產預測
prod_predictions_proba = model.predict_proba(X_prod)[:, 1]

print("模擬了新的生產環境數據，並取得了模型預測結果。")

# --- Step 3: 監控：比較數據與預測分佈 ---
print("\n[Step 3] 監控：比較數據與預測分佈...")

# 3.1 監控輸入特徵：年齡分佈比較
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_train['age'], color='skyblue', kde=True, stat='density', label='訓練資料 Age', alpha=0.6)
sns.histplot(df_prod['age'], color='red', kde=True, stat='density', label='生產資料 Age', alpha=0.6)
plt.title('年齡分佈比較 (訓練 vs. 生產)')
plt.xlabel('年齡')
plt.ylabel('密度')
plt.legend()

# 3.2 監控輸入特徵：收入分佈比較
plt.subplot(1, 2, 2)
sns.histplot(df_train['income'], color='lightgreen', kde=True, stat='density', label='訓練資料 Income', alpha=0.6)
sns.histplot(df_prod['income'], color='orange', kde=True, stat='density', label='生產資料 Income', alpha=0.6)
plt.title('收入分佈比較 (訓練 vs. 生產)')
plt.xlabel('收入')
plt.ylabel('密度')
plt.legend()

plt.tight_layout()
plt.show()

# 3.3 監控模型輸出：預測機率分佈比較
plt.figure(figsize=(8, 5))
sns.histplot(train_predictions_proba, color='skyblue', kde=True, stat='density', label='訓練時預測機率', alpha=0.6)
sns.histplot(prod_predictions_proba, color='red', kde=True, stat='density', label='生產時預測機率', alpha=0.6)
plt.title('模型預測機率分佈比較 (訓練 vs. 生產)')
plt.xlabel('點擊機率')
plt.ylabel('密度')
plt.legend()
plt.show()

print("\n--- 監控結果分析 ---")
print("從圖表中可以看到：")
print("1. 年齡分佈：生產環境中的客戶年齡明顯變年輕了，這就是『數據漂移』！")
print("2. 收入分佈：生產環境中的客戶收入分佈變得更廣，這也是『數據漂移』！")
print("3. 預測機率分佈：由於輸入數據的變化，模型給出的預測機率分佈也發生了改變。")
print("這些變化都可能導致模型在生產環境中的表現不如預期。")

print("\n--- 恭喜你，完成了簡易模型監控的建置！ ---")
```

#### 3. 程式碼解析與學習點

1.  **模擬訓練與生產資料：** 我們用 `numpy` 隨機生成了兩組數據，分別代表模型訓練時的資料和部署後在生產環境中遇到的新資料。
2.  **模擬數據漂移：** 在生成生產資料時，我們故意讓 `age` 的平均值變小，讓 `income` 的標準差變大。這樣，當我們比較訓練和生產資料的分佈時，就能清楚地看到「漂移」發生了。
3.  **訓練模型：** 使用 `LogisticRegression` 訓練了一個簡單的分類模型。
4.  **監控輸入數據分佈：** 我們使用 `seaborn.histplot` 來繪製 `age` 和 `income` 在訓練時和生產時的直方圖。如果兩個直方圖的形狀、中心點或寬度有明顯差異，就說明輸入數據發生了漂移。
5.  **監控輸出預測分佈：** 同樣地，我們比較了模型在訓練資料上預測的機率分佈和在生產資料上預測的機率分佈。如果預測的機率分佈發生了變化，通常意味著模型正在面對它不熟悉的數據，或者數據的底層關係已經改變。
6.  **視覺化：** `matplotlib` 和 `seaborn` 是你進行數據監控和可觀測性的好幫手，直觀的圖表能幫助你快速發現問題。

#### 4. 下一步呢？

今天的範例是監控最基礎的**數據漂移 (Data Drift)** 和**預測漂移 (Prediction Drift)**。這已經是一個非常好的開始了！在實際的 MLOps 場景中，你會進一步考慮：

*   **性能監控 (Performance Monitoring)**：當你有真實標籤時（通常會延遲獲得），你可以計算模型在生產環境中的準確度、F1 分數、RMSE 等指標。
*   **異常檢測 (Anomaly Detection)**：監控是否有極端的異常值輸入，或者模型輸出了異常的預測。
*   **可解釋性 (Explainability)**：當模型表現下降時，能否知道是哪個特徵或哪組特徵導致了問題？（例如 LIME, SHAP 等工具）
*   **自動化警報 (Automated Alerting)**：當任何監控指標超出預設閾值時，自動發送通知（Email, Slack 等）。
*   **專業工具：** 學習使用更專業的 MLOps 監控工具，例如：
    *   **Evidently AI / whylogs:** 專注於數據和模型漂移檢測。
    *   **MLflow:** 除了實驗追蹤，也能用於模型部署和簡單的監控。
    *   **Prometheus / Grafana:** 通用的監控和視覺化工具，可以整合到 MLOps 管線中。
    *   **雲端服務:** AWS Sagemaker Model Monitor, Azure ML Monitoring 等，都提供了強大的託管監控解決方案。

#### 結語

恭喜你！今天你不僅理解了 MLOps 模型監控與可觀測性的重要性，還親手搭建了一個簡易的監控系統。這是一個非常重要的里程碑，因為它讓你的 AI 模型從一個「實驗室產品」轉變為一個「健康運作的服務」。

永遠記住，MLOps 的核心就是將 AI 從實驗室帶入真實世界，並確保它在那裡也能茁壯成長！繼續探索，你會發現 MLOps 的世界充滿了無限可能！我們下次見！