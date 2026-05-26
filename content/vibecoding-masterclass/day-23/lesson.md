好的，我的朋友！歡迎來到我們精彩的程式學習旅程！

---

## 第 23 天：建立互動式數據儀表板

嘿，夥伴們！今天我們將迎來一個超級令人興奮的主題：**建立互動式數據儀表板**！想像一下，你的數據不再是冷冰冰的表格或靜態圖片，而是可以動起來，讓使用者自由探索、篩選，甚至講述一個動態的故事！這聽起來是不是很酷？

在現實世界中，從商業決策、科學研究到日常數據分析，互動式儀表板都是一個不可或缺的工具。它能幫助我們更快地從數據中發現洞察，做出更好的決策。而今天，我們的秘密武器就是一個叫做 **Streamlit** 的 Python 函式庫！它超級容易上手，讓你可以用純 Python 程式碼快速建立出色的網路應用和儀表板。

準備好了嗎？讓我們一起將數據活化起來！

### 目標

*   了解為什麼需要互動式儀表板。
*   使用 Streamlit 建立一個基礎的儀表板。
*   加入互動式元素，如選擇框 (selectbox) 和滑動條 (slider)。
*   利用 Plotly Express 進行數據視覺化。

### 前置準備

在開始之前，請確保你已經安裝了以下函式庫。如果你還沒安裝，打開你的終端機或命令提示字元，輸入：

```bash
pip install streamlit pandas plotly_express numpy
```

### 步驟一：準備數據

為了讓我們的儀表板有內容，我們需要一些數據。這次我們不讀取外部文件，而是直接用 `pandas` 和 `numpy` 創造一些模擬數據。這會讓我們的範例更完整、更容易運行。

```python
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# 1. 模擬一些數據
@st.cache_data # 使用 Streamlit 的緩存功能，讓數據只生成一次
def generate_data():
    np.random.seed(42)
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    products = ['A', 'B', 'C', 'D']
    regions = ['East', 'West', 'North', 'South']

    data = {
        'Date': np.random.choice(dates, 500),
        'Product': np.random.choice(products, 500),
        'Region': np.random.choice(regions, 500),
        'Sales': np.random.randint(100, 1000, 500)
    }
    df = pd.DataFrame(data)
    return df

df = generate_data()

# 在儀表板側邊欄顯示應用標題
st.sidebar.title("我的互動式銷售儀表板")
```

這段程式碼會生成一個包含日期、產品、地區和銷售額的數據框。我們還使用 `st.cache_data`，這是一個 Streamlit 的魔術，它會確保我們的數據只生成一次，而不是每次互動都重新生成，這樣能大大提升性能！

### 步驟二：你的第一個 Streamlit 應用

接下來，我們來建立 Streamlit 的基本結構。

在你的 Python 檔案（例如 `dashboard_app.py`）中，加入以下程式碼：

```python
# ... (承接上面的數據生成程式碼) ...

st.title("銷售數據分析")
st.write("這是一個簡單的互動式儀表板，用來分析我們的模擬銷售數據。")

# 顯示原始數據框 (非必要，但對初學者了解數據很有幫助)
st.subheader("原始數據預覽")
st.dataframe(df.head()) # 只顯示前幾行
```

**如何運行你的儀表板？**

保存你的檔案為 `dashboard_app.py`，然後在終端機中導航到該檔案所在的目錄，輸入：

```bash
streamlit run dashboard_app.py
```

這將會自動在你的瀏覽器中打開一個新的頁面，你應該能看到你的標題、說明和數據預覽！是不是很酷？

### 步驟三：加入互動元素 - 篩選器！

現在，我們來加入讓儀表板「動」起來的核心部分：互動式篩選器！

#### 1. 地區選擇框 (Selectbox)

我們將在側邊欄添加一個選擇框，讓使用者可以選擇查看哪個地區的數據。

```python
# ... (承接上面的程式碼) ...

# 地區篩選器
selected_region = st.sidebar.selectbox(
    "選擇地區:",
    options=['所有地區'] + list(df['Region'].unique())
)

# 根據選擇的地區篩選數據
filtered_df = df.copy() # 先複製一份原始數據，以免影響原始數據

if selected_region != '所有地區':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

# 顯示篩選後的數據預覽
st.subheader(f"篩選後的數據 ({selected_region})")
st.dataframe(filtered_df.head())
```

現在刷新你的瀏覽器，你會看到一個側邊欄，裡面有一個下拉式選單。試著選擇不同的地區，你會發現下面的數據預覽也跟著變化了！

#### 2. 銷售額範圍滑動條 (Slider)

再添加一個滑動條，讓使用者可以設定銷售額的上下限。

```python
# ... (承接上面的程式碼) ...

# 銷售額範圍篩選器
min_sales = int(df['Sales'].min())
max_sales = int(df['Sales'].max())

sales_range = st.sidebar.slider(
    "選擇銷售額範圍:",
    min_value=min_sales,
    max_value=max_sales,
    value=(min_sales, max_sales) # 預設值為整個範圍
)

# 根據銷售額範圍再次篩選數據
filtered_df = filtered_df[
    (filtered_df['Sales'] >= sales_range[0]) &
    (filtered_df['Sales'] <= sales_range[1])
]

# 顯示最終篩選後的數據總覽
st.subheader("最終篩選數據總覽")
st.write(f"顯示了 {len(filtered_df)} 筆數據。")
st.dataframe(filtered_df.head())
```

現在你的側邊欄有了兩個互動元素！拖動滑動條，數據預覽會根據你設定的銷售額範圍再次更新。太棒了！

### 步驟四：數據視覺化

僅僅顯示表格可能還不夠直觀，讓我們來添加一些漂亮的圖表！我們將使用 `plotly.express`，它能輕鬆地創建互動式圖表。

讓我們來繪製一個長條圖，顯示不同產品的總銷售額：

```python
# ... (承接上面的程式碼) ...

st.subheader("各產品銷售總額")

if not filtered_df.empty:
    product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()
    fig = px.bar(
        product_sales,
        x='Product',
        y='Sales',
        title='各產品銷售總額',
        color='Product' # 讓每個產品有不同的顏色
    )
    st.plotly_chart(fig, use_container_width=True) # 顯示 Plotly 圖表，並讓它佔滿容器寬度
else:
    st.warning("沒有數據符合篩選條件，請調整篩選器。")

# 也可以再加一個時間趨勢圖
st.subheader("每日銷售趨勢")
if not filtered_df.empty:
    daily_sales = filtered_df.groupby('Date')['Sales'].sum().reset_index()
    fig_line = px.line(
        daily_sales,
        x='Date',
        y='Sales',
        title='每日銷售趨勢',
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning("沒有數據符合篩選條件，請調整篩選器。")
```

再次刷新你的瀏覽器！現在，你看到的不再僅僅是數據表格，而是一個隨著你的篩選條件動態變化的長條圖和折線圖！點擊圖表中的圖例，你甚至可以隱藏或顯示特定的產品或日期數據。這就是互動式儀表板的魅力！

### 完整的程式碼範例

將所有程式碼合併到一個檔案 `dashboard_app.py` 中：

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. 模擬一些數據 (使用 Streamlit 的緩存功能)
@st.cache_data
def generate_data():
    np.random.seed(42) # 為了確保每次運行數據都一樣
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100))
    products = ['Apple', 'Banana', 'Cherry', 'Date']
    regions = ['East', 'West', 'North', 'South']

    data = {
        'Date': np.random.choice(dates, 500),
        'Product': np.random.choice(products, 500),
        'Region': np.random.choice(regions, 500),
        'Sales': np.random.randint(100, 1000, 500)
    }
    df = pd.DataFrame(data)
    return df

df = generate_data()

# 應用標題
st.sidebar.title("我的互動式銷售儀表板")
st.title("銷售數據分析")
st.write("這是一個簡單的互動式儀表板，用來分析我們的模擬銷售數據。")

# --- 側邊欄篩選器 ---
st.sidebar.header("篩選條件")

# 地區篩選器
selected_region = st.sidebar.selectbox(
    "選擇地區:",
    options=['所有地區'] + sorted(df['Region'].unique().tolist()) # 排序並加入'所有地區'選項
)

# 銷售額範圍篩選器
min_sales_overall = int(df['Sales'].min())
max_sales_overall = int(df['Sales'].max())

sales_range = st.sidebar.slider(
    "選擇銷售額範圍:",
    min_value=min_sales_overall,
    max_value=max_sales_overall,
    value=(min_sales_overall, max_sales_overall) # 預設值為整個範圍
)

# --- 數據篩選邏輯 ---
filtered_df = df.copy()

if selected_region != '所有地區':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

filtered_df = filtered_df[
    (filtered_df['Sales'] >= sales_range[0]) &
    (filtered_df['Sales'] <= sales_range[1])
]

# --- 主區域顯示內容 ---

st.subheader("篩選結果總覽")
if not filtered_df.empty:
    st.write(f"當前顯示 **{len(filtered_df)}** 筆數據。")
    st.dataframe(filtered_df.head(10)) # 顯示前10行數據
else:
    st.warning("沒有數據符合當前篩選條件，請調整篩選器。")

# --- 數據視覺化 ---
if not filtered_df.empty:
    st.markdown("---") # 分隔線

    st.subheader("各產品銷售總額分析")
    product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()
    fig_bar = px.bar(
        product_sales,
        x='Product',
        y='Sales',
        title='各產品總銷售額',
        color='Product',
        labels={'Product': '產品名稱', 'Sales': '總銷售額'},
        template='plotly_white' # 選擇一個好看的模板
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("每日銷售趨勢分析")
    # 確保日期是datetime類型，並排序
    daily_sales = filtered_df.groupby('Date')['Sales'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date') # 按日期排序
    fig_line = px.line(
        daily_sales,
        x='Date',
        y='Sales',
        title='每日銷售趨勢',
        markers=True, # 顯示每個數據點
        labels={'Date': '日期', 'Sales': '總銷售額'},
        template='plotly_white'
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.markdown("---")
    st.info("請調整篩選條件以查看圖表。")

st.sidebar.markdown("---")
st.sidebar.info("這是使用 Streamlit 建立的簡單儀表板！")

```

### 總結與下一步

恭喜你！在今天的課程中，你已經成功建立了一個功能強大、互動性十足的數據儀表板！你學會了：

*   使用 Streamlit 快速搭建一個網頁應用。
*   創建模擬數據來填充你的儀表板。
*   添加下拉選單 (selectbox) 和滑動條 (slider) 這些互動元素。
*   利用 Plotly Express 製作漂亮的互動式圖表。

這只是一個開始！互動式儀表板的世界廣闊而精彩。

**你可以繼續探索：**

*   **更多互動元素：** Streamlit 還有按鈕 (`st.button`)、多選框 (`st.multiselect`)、日期輸入 (`st.date_input`) 等等。
*   **更多圖表類型：** 嘗試使用 `plotly.express` 繪製散點圖、熱力圖或其他你喜歡的圖表。
*   **佈局優化：** 使用 `st.columns()` 創建多列佈局，讓你的儀表板更美觀。
*   **真實數據：** 嘗試從 CSV、Excel 或資料庫中載入你自己的數據。
*   **部署：** 學習如何將你的 Streamlit 應用部署到網路上，讓其他人也能看到！

繼續保持這份好奇心和學習熱情！你的數據故事才剛剛開始！明天我們將會探討更多精彩的內容！加油！