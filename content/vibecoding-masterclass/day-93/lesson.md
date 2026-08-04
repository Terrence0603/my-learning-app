哈囉，我的未來 AI 大師們！

不知不覺，我們已經走到了 MLOps 學習旅程的第 93 天！如果你一直跟著我的步伐，現在應該已經能訓練、評估，甚至部署你的機器學習模型了。是不是很有成就感呢？

但等等，你可能覺得模型訓練完、部署了就萬事大吉？可惜，現實世界可不是這樣運作的！想像一下，你辛辛苦苦訓練出的模型，在生產環境中悄悄地表現變差了，但你卻不知道！這就像你的愛車引擎燈亮了，但你卻沒發現，直到它拋錨在半路上。

這就是為什麼 MLOps 中有兩個超級重要的概念：**模型監控 (Model Monitoring)** 和 **漂移偵測 (Drift Detection)**。今天，我們就要像偵探一樣，學會如何讓模型不只會工作，還要工作得「好」，並且在「變壞」之前就發現它！

---

### 🚀 為什麼模型監控如此重要？

模型監控就像是模型的健康檢查報告，它持續追蹤模型在生產環境中的各種指標，例如：
*   **預測結果的分佈**：模型是不是開始瘋狂預測某個類別？
*   **輸入數據的特徵分佈**：是不是有新的數據湧入，和模型訓練時見過的數據很不一樣？
*   **模型的性能**：如果我們有真實標籤，模型現在的準確度、召回率、F1 分數還好嗎？

透過監控，我們可以在模型表現惡化之前就收到警報，及時介入處理。

---

### 🔍 什麼是模型漂移 (Drift)？

而「漂移 (Drift)」就是這個健康檢查中最需要警惕的信號之一。它指的是模型在生產環境中的輸入數據或其與目標變量的關係發生了變化。主要有兩種：

1.  **數據漂移 (Data Drift)**：
    *   **定義**：輸入模型的數據本身的統計特性發生了變化。
    *   **例子**：你的模型是預測房價的，它用的是 2020 年的經濟數據訓練。但現在是 2024 年，通貨膨脹、利率、市場偏好都變了，新來的房產數據和 2020 年的數據分佈完全不同了。
2.  **概念漂移 (Concept Drift)**：
    *   **定義**：輸入數據和目標變量之間的關係發生了變化，即使輸入數據本身可能沒有顯著變化。
    *   **例子**：你的模型預測客戶是否會點擊廣告。由於節日促銷活動或新的營銷策略，即使是相同的客戶群體，他們點擊廣告的意願和模式也改變了。模型原有的「概念」不再適用了。

無論是哪種漂移，結果都一樣：模型預測的準確度會下降，影響業務決策！

---

### 🛠️ 實戰：使用 `evidently AI` 進行漂移偵測

要偵測這些「不速之客」，我們需要一些趁手的工具。今天，我們要介紹一個非常棒的開源庫：`evidently AI`。它能幫我們快速生成詳細的報告，一眼看出數據和模型表現的變化。

首先，你需要安裝它：

```bash
pip install evidently pandas scikit-learn
```

接下來，讓我們來看看如何用程式碼實現：

```python
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("--- MLOps 模型監控與漂移偵測實戰 ---")

# --- 1. 模擬數據 ---
# 假設我們有一個簡單的二元分類問題 (例如：客戶流失預測)
# 數據包含 'feature_1', 'feature_2' 和 'target' (0 或 1)

# 參考數據 (Reference Data - 想像這是模型訓練時的數據)
np.random.seed(42)
ref_data_size = 1000
reference_data = pd.DataFrame({
    'feature_1': np.random.normal(50, 10, ref_data_size),
    'feature_2': np.random.normal(100, 20, ref_data_size),
    'categorical_feature': np.random.choice(['A', 'B', 'C'], ref_data_size),
    'target': np.random.randint(0, 2, ref_data_size)
})
# 為了讓參考數據更像真實情況，讓 target 和 feature_1 有點關係
reference_data['target'] = (reference_data['feature_1'] > 55).astype(int)
print(f"參考數據形狀: {reference_data.shape}")

# 當前數據 (Current Data - 想像這是模型現在正在處理的數據)
# 我們故意引入一些漂移 (Data Drift) 和 (Concept Drift 模擬)
current_data_size = 500
current_data = pd.DataFrame({
    # feature_1 的均值發生了變化 (數據漂移)
    'feature_1': np.random.normal(60, 12, current_data_size),
    'feature_2': np.random.normal(102, 21, current_data_size), # 略微變化
    'categorical_feature': np.random.choice(['A', 'B', 'D'], current_data_size), # 新增了类别 'D'
    'target': np.random.randint(0, 2, current_data_size)
})
# 概念漂移: 現在 target 和 feature_2 關係更大了
current_data['target'] = (current_data['feature_2'] > 110).astype(int)
print(f"當前數據形狀: {current_data.shape}\n")


# --- 2. 訓練一個簡單的模型 (為了展示性能監控) ---
# 將參考數據分成訓練和測試集
X_ref = reference_data[['feature_1', 'feature_2', 'categorical_feature']]
y_ref = reference_data['target']

# 這裡需要對類別特徵進行 One-Hot 編碼
X_ref = pd.get_dummies(X_ref, columns=['categorical_feature'], drop_first=True)

model = RandomForestClassifier(random_state=42)
model.fit(X_ref, y_ref)

# 在 'current_data' 上進行預測
X_current = current_data[['feature_1', 'feature_2', 'categorical_feature']]
X_current = pd.get_dummies(X_current, columns=['categorical_feature'], drop_first=True)

# 確保 'current_data' 和 'reference_data' 的特徵列匹配
# 這裡處理新的類別 'D'，並補齊 reference_data 中沒有的列
missing_cols_in_current = set(X_ref.columns) - set(X_current.columns)
for c in missing_cols_in_current:
    X_current[c] = 0
X_current = X_current[X_ref.columns] # 確保列的順序也一致

current_preds = model.predict(X_current)
current_pred_proba = model.predict_proba(X_current)

# 將預測結果加回 current_data DataFrame，以便 evidently 監控
current_data['prediction'] = current_preds
current_data['pred_proba_0'] = current_pred_proba[:, 0]
current_data['pred_proba_1'] = current_pred_proba[:, 1]


# --- 3. 創建 Evidently 報告 ---
# 定義 column_mapping，告訴 Evidently 哪些是特徵，哪些是目標
column_mapping = {
    'target': 'target',
    'prediction': 'prediction',
    'numerical_features': ['feature_1', 'feature_2'],
    'categorical_features': ['categorical_feature'],
    'datetime_features': None # 如果有時間戳特徵可以放在這裡
}

# 初始化一個 Evidently 報告
# DataDriftPreset: 用於偵測數據漂移
# ClassificationPreset: 用於評估分類模型性能 (需要有真實標籤)
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset(probas=['pred_proba_0', 'pred_proba_1']) # 如果是二元分類的機率
])

print("正在生成 Evidently 報告，這可能需要一些時間...\n")

# --- 4. 運行報告 ---
data_drift_report.run(
    reference_data=reference_data,
    current_data=current_data,
    column_mapping=column_mapping
)

# --- 5. 顯示或保存報告 ---
# 在 Jupyter Notebook 或 Colab 中會直接顯示互動式報告
data_drift_report.show()

# 你也可以將報告保存為 HTML 文件，在瀏覽器中打開
data_drift_report.save_html("mlops_drift_detection_report.html")
print("報告已生成並保存為 'mlops_drift_detection_report.html'")

print("\n--- 實戰結束，請查看生成的報告！ ---")
```

---

### 📊 解讀你的 Evidently 報告

當你運行上面的程式碼，`evidently` 會在你的 Jupyter Notebook 或瀏覽器中生成一個互動式的 HTML 報告。

這個報告會非常詳細地列出：

*   **數據漂移總覽 (Data Drift Overview)**：會告訴你總共有多少特徵發生了漂移，以及漂移的嚴重程度。你應該會看到因為 `feature_1` 均值變化和 `categorical_feature` 新類別而導致的漂移警告。
*   **每個特徵的詳細分析**：對於 `feature_1` 和 `feature_2` 這樣的數值特徵，它會顯示分佈圖（直方圖或 KED 圖），並比較參考數據和當前數據的分佈差異，以及統計檢定結果 (p-value)。對於類別特徵，它會顯示每個類別的頻率變化。
*   **模型性能監控 (Classification Performance)**：你會看到模型的準確度、精確度、召回率、F1 分數等指標，並會比較它們在參考數據和當前數據上的表現。由於我們模擬了概念漂移，你很可能會看到性能下降的跡象！
*   **目標分佈變化**：`target` 變量本身的分佈是否發生了變化，這也是概念漂移的一個重要信號。

**下一步呢？**

如果報告中出現「Drift detected (漂移偵測)」的標示，特別是多個特徵或關鍵特徵出現漂移，這就是一個警報！這意味著你的模型可能正在用它不熟悉的數據進行預測，或者它學到的「規則」已經不再適用了。

此時，你可能需要：
1.  **深入調查**：是哪個特徵在漂移？漂移的原因是什麼？是數據採集問題、外部環境變化還是用戶行為改變？
2.  **重新訓練模型**：使用更新、更符合當前數據分佈的數據集來重新訓練你的模型。
3.  **調整模型**：考慮使用更具適應性的模型，或實施線上學習策略。

---

### 💡 總結與展望

恭喜你！今天我們一起解鎖了 MLOps 中一個非常實用且關鍵的技能！模型監控和漂移偵測不是可選項，而是保證 AI 系統穩定和高效運行的「保險絲」和「指南針」。

透過 `evidently` 這樣強大的工具，即使是初學者也能快速上手，為你的模型建立一道堅固的防線。記住，成功的 AI 專案不僅僅是建立一個好模型，更是要讓這個模型在變幻莫測的現實世界中持續保持「最佳狀態」！

多練習，你會越來越有感覺的！我們下個階段再見，繼續探索 MLOps 的奧秘！