恭喜你，來到【第 30 天】的學習旅程！這是一個重要的里程碑，代表你已經從一個完全的初學者，成長為一位對機器學習有著紮實理解的學習者。今天，我們將觸及一個在實際生產環境中極為關鍵，卻常常被初學者忽略的主題：**模型監控與數據漂移偵測**。

想像一下，你花費了大量心力，訓練出了一隻超級聰明、能精準預測的 AI 狗狗。你把它部署到了真實世界，讓它為你工作。但問題來了：你會不會就此撒手不管，認為它會永遠都那麼聰明、那麼準確呢？當然不會！環境會變、狗狗會老、新的狀況會出現。你的 AI 模型也是一樣！

---

## 主題：第 30 天：生產環境下模型監控與數據漂移偵測

### 為什麼模型需要監控？

在部署模型之後，我們必須持續追蹤它的表現，這就是 **模型監控 (Model Monitoring)**。為什麼呢？因為現實世界是動態變化的！

1.  **效能下降 (Performance Degradation)**：模型在訓練時表現出色，但在實際應用中，可能會因為各種原因導致預測準確率降低。
2.  **商業影響 (Business Impact)**：不準確的模型可能導致錯誤決策，造成金錢損失或客戶不滿。
3.  **發現問題 (Problem Detection)**：監控可以幫助我們及早發現數據管道問題、模型 bugs 或其他系統異常。

### 什麼是數據漂移 (Data Drift)？

在眾多導致模型效能下降的原因中，**數據漂移 (Data Drift)** 是最常見也最難以避免的一種。

想像你的模型是根據「夏天」的交通模式訓練出來的，它知道夏天時大家開車習慣。但如果現在是「冬天」，路面結冰，車速變慢，路況變得完全不同了，你的「夏天模型」還能準確預測嗎？顯然不能！這就是數據漂移：**輸入給模型的數據特徵，在一段時間後，其統計特性發生了變化，不再符合模型訓練時所見的數據模式。**

數據漂移可能是由於：
*   **真實世界變化**：經濟波動、消費者行為改變、流行趨勢轉換。
*   **數據採集系統變化**：新的感測器、不同的數據源、數據前處理流程調整。

當數據漂移發生時，你的模型會開始「說胡話」，預測結果變得不準確，甚至會產生完全錯誤的判斷。

### 如何監控模型與偵測數據漂移？

核心思想就是：**比較「過去」和「現在」**。

1.  **模型效能指標 (Model Performance Metrics)**：
    *   對於分類模型：準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall)、F1 分數。
    *   對於迴歸模型：均方誤差 (MSE)、平均絕對誤差 (MAE)。
    *   **如何計算？** 需要有真實標籤 (Ground Truth)。如果你的系統能很快得到真實標籤，就可以直接計算。

2.  **數據特性指標 (Data Quality & Feature Metrics)**：
    *   **輸入特徵的分佈變化**：比較每個輸入特徵（例如，客戶年齡、產品價格）在訓練集（參考數據）和生產環境中（當前數據）的平均值、中位數、標準差、分佈形狀。
    *   **缺失值比例變化**：是否有新的特徵開始出現大量的缺失值？
    *   **數據範圍變化**：某個特徵的值是否超出了訓練時的合理範圍？

### 實戰範例：使用 Evidently AI 偵測數據漂移

有許多工具可以幫助我們進行模型監控，例如 MLflow、Arize、whylogs 等。今天，我們將使用一個輕量級且功能強大的開源工具 **Evidently AI** 來展示數據漂移的偵測。

**Evidently AI** 可以生成交互式的 HTML 報告，非常直觀地展示數據的各種統計特性變化。

#### 步驟 1：安裝 Evidently AI

在你的終端機或 Jupyter Notebook 中執行：

```bash
pip install evidently pandas scikit-learn numpy
```

#### 步驟 2：準備參考數據 (Reference Data) 和當前數據 (Current Data)

我們來模擬一下訓練數據 (參考數據) 和在生產環境中收集到的新數據 (當前數據)。為了演示，我們特意在當前數據中引入一些漂移。

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# --- 1. 生成模擬數據 ---
# 假設我們有一個簡單的二元分類問題
X_ref, y_ref = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    random_state=42
)
ref_df = pd.DataFrame(X_ref, columns=[f'feature_{i}' for i in range(10)])
ref_df['target'] = y_ref
ref_df['prediction'] = np.where(ref_df['feature_0'] > 0, 1, 0) # 簡化預測，只是為了有預測欄位

# 模擬生產環境中的當前數據
X_curr, y_curr = make_classification(
    n_samples=500, # 假設當前數據量較少
    n_features=10,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    random_state=100
)
curr_df = pd.DataFrame(X_curr, columns=[f'feature_{i}' for i in range(10)])
curr_df['target'] = y_curr
curr_df['prediction'] = np.where(curr_df['feature_0'] > 0.5, 1, 0) # 簡化預測

# --- 2. 引入數據漂移 ---
# 我們故意讓 feature_0 和 feature_1 的分佈在當前數據中發生變化
curr_df['feature_0'] = curr_df['feature_0'] * 1.5 + 0.5 # 改變均值和方差
curr_df['feature_1'] = curr_df['feature_1'] - 1.0       # 改變均值
curr_df['feature_2'] = curr_df['feature_2'] * 0.8       # 改變均值和方差

print("參考數據（訓練時）：")
print(ref_df.head())
print("\n當前數據（生產環境）：")
print(curr_df.head())
```

#### 步驟 3：運行 Evidently 數據漂移報告

現在，我們可以使用 `DataDriftPreset` 來生成一個數據漂移報告：

```python
# 建立一個 Evidently 報告對象，並指定我們想要監控的指標集
data_drift_report = Report(metrics=[
    DataDriftPreset(), # 偵測數據漂移的核心預設集
])

# 運行報告，傳入參考數據和當前數據
print("\n正在生成數據漂移報告，請稍候...")
data_drift_report.run(
    reference_data=ref_df,
    current_data=curr_df,
    column_mapping=None # 如果你的數據有不同的欄位名稱，可以在這裡映射
)

# 將報告保存為 HTML 文件
report_path = "data_drift_report.html"
data_drift_report.save_html(report_path)
print(f"數據漂移報告已保存至：{report_path}")

# 在 Jupyter 環境中，你也可以直接顯示報告
# data_drift_report.show()
```

#### 步驟 4：查看報告並解釋結果

打開生成的 `data_drift_report.html` 文件，你將會看到一個豐富的交互式報告。

報告會包含：
*   **總體數據漂移概覽**：哪些特徵發生了漂移，漂移的程度。
*   **每個特徵的詳細分析**：
    *   參考數據和當前數據的分佈直方圖或密度圖，你可以直觀地看到分佈的差異。
    *   統計檢定結果 (例如 Kolmogorov-Smirnov test 或 chi-square test)，顯示漂移的統計顯著性。
    *   缺失值、數據類型等基本統計信息對比。

你會發現在 `feature_0`、`feature_1` 和 `feature_2` 這幾個我們故意引入漂移的特徵上，Evidently AI 會明確地標示出「Drift Detected」！很酷吧！

### 當偵測到漂移或效能下降時該怎麼辦？

別慌！這是一個正常現象，也是模型監控的價值所在。當你收到監控警報時，可以考慮以下步驟：

1.  **深入分析**：查看漂移報告，找出是哪些特徵發生了漂移，以及漂移的嚴重程度。
2.  **調查原因**：數據漂移的原因是什麼？是真實世界的變化，還是數據管道出了問題？
3.  **重新訓練模型**：最常見的解決方案是使用最新的、包含漂移數據的新數據集來重新訓練你的模型。這能讓模型適應新的數據分佈。
4.  **調整特徵工程**：某些情況下，可能需要調整特徵工程策略，以更好地處理變化的數據。
5.  **回滾模型**：如果新模型表現不佳或問題緊急，可能需要暫時回滾到以前穩定運行的模型版本。

---

### 總結與展望

恭喜你完成了這段旅程的第 30 天！你現在不僅知道如何訓練和部署模型，更理解了在模型投入使用後，持續監控其「健康狀況」有多麼重要。模型監控與數據漂移偵測不是一個可選的步驟，而是你成為優秀 ML 工程師的必經之路。

這只是冰山一角。生產環境下的機器學習 (MLOps) 是一個廣闊的領域，包含版本控制、CI/CD、資源管理、可解釋性 (XAI) 等等。但今天你掌握的「監控」技能，是所有這些進階概念的基石。

繼續探索，保持好奇！你的機器學習之旅才剛剛開始，未來還有更多令人興奮的挑戰和發現等著你！加油！