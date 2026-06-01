太棒了！歡迎來到我們為期 30 天程式學習之旅的第 29 天！🎉 經過這麼多天的學習，你已經累積了扎實的程式基礎，並且能夠逐步建構出令人驚豔的應用。今天，我們將迎來一個令人興奮的里程碑：**整合預測 API 至互動式儀表板！**

想像一下，你不再只是看到靜態的數據，而是可以即時地與你的數據互動，並且看到由預測模型所產生的智慧結果！這就是我們今天要實現的目標，準備好讓你的專案「活」起來了嗎？

### 為什麼要整合預測 API？

在實際應用中，我們常常需要利用機器學習模型來預測未來的趨勢、分類數據，或是提供個人化的建議。而 API (Application Programming Interface) 就像是一個橋樑，讓我們的應用程式能夠輕鬆地呼叫這些預測模型的「智慧」。

透過將預測 API 整合到互動式儀表板中，我們可以：

*   **即時洞察：** 不僅看到過去的數據，還能預見未來可能的走向。
*   **提升互動性：** 使用者可以透過儀表板的控制項，輸入參數，並立即看到預測結果的變化，提供更豐富的體驗。
*   **賦能決策：** 幫助使用者基於預測數據做出更明智的決策。

### 我們要建構什麼？

今天，我們將以一個簡單的場景為例。假設我們有一個預測房價的 API，它可以根據房屋的面積（square footage）來預測房價。我們將建立一個簡單的網頁儀表板，讓使用者輸入房屋面積，然後透過呼叫這個預測 API，將預測的房價顯示在儀表板上。

### 所需工具

*   **前端技術：** HTML, CSS, JavaScript (用於建立儀表板介面)
*   **API 呼叫：** JavaScript 的 `fetch` API (用於與預測 API 互動)
*   **預測 API (模擬)：** 為了方便教學，我們將模擬一個預測 API。在真實世界中，這可能是一個部署好的機器學習模型服務。

### 程式碼範例

#### 1. HTML 結構 (index.html)

首先，我們建立一個簡單的 HTML 檔案來架構我們的儀表板。

```html
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>房價預測儀表板</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="dashboard-container">
        <h1>互動式房價預測儀表板</h1>

        <div class="input-section">
            <label for="squareFeet">房屋面積 (平方英尺):</label>
            <input type="number" id="squareFeet" placeholder="請輸入面積">
            <button id="predictButton">預測房價</button>
        </div>

        <div class="output-section">
            <h2>預測結果</h2>
            <p id="predictedPrice">等待輸入與預測...</p>
        </div>
    </div>

    <script src="script.js"></script>
</body>
</html>
```

*   我們有一個輸入框 (`input-section`)，讓使用者輸入房屋面積。
*   一個按鈕 (`predictButton`)，觸發預測。
*   一個區域 (`output-section`)，顯示預測的房價。

#### 2. CSS 樣式 (style.css) (可選，但建議美化一下！)

為了讓儀表板看起來更專業，我們可以加上一些 CSS。

```css
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f4f7f6;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
}

.dashboard-container {
    background-color: #ffffff;
    padding: 30px 40px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    text-align: center;
    width: 90%;
    max-width: 600px;
}

h1 {
    color: #333;
    margin-bottom: 30px;
}

.input-section {
    margin-bottom: 30px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}

.input-section label {
    font-weight: bold;
    color: #555;
}

.input-section input[type="number"] {
    padding: 12px;
    border: 1px solid #ccc;
    border-radius: 5px;
    font-size: 1rem;
    width: 60%;
    outline: none;
}

.input-section button {
    padding: 12px 25px;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.input-section button:hover {
    background-color: #0056b3;
}

.output-section {
    margin-top: 30px;
    border-top: 1px solid #eee;
    padding-top: 20px;
}

.output-section h2 {
    color: #333;
    margin-bottom: 15px;
}

#predictedPrice {
    font-size: 1.5rem;
    color: #28a745; /* 綠色表示成功 */
    font-weight: bold;
}
```

#### 3. JavaScript 邏輯 (script.js)

這就是我們整合 API 的核心！

```javascript
// 模擬一個預測 API 的 URL
// 在真實情況下，這會是你實際的 API 端點
const PREDICTION_API_URL = 'YOUR_PREDICTION_API_ENDPOINT'; // 替換成你的 API 網址

const squareFeetInput = document.getElementById('squareFeet');
const predictButton = document.getElementById('predictButton');
const predictedPriceDisplay = document.getElementById('predictedPrice');

// 模擬的預測函數，用於本地開發測試 (如果 API 還沒準備好)
// 在真實情況下，你會直接呼叫 PREDICTION_API_URL
async function simulatePrediction(squareFeet) {
    // 簡單的線性預測模擬: 房價 = 100 * 面積 + 50000
    await new Promise(resolve => setTimeout(resolve, 500)); // 模擬網路延遲
    return 100 * squareFeet + 50000;
}

predictButton.addEventListener('click', async () => {
    const squareFeet = squareFeetInput.value;

    if (!squareFeet) {
        predictedPriceDisplay.textContent = '請輸入房屋面積！';
        predictedPriceDisplay.style.color = '#dc3545'; // 顯示紅色錯誤訊息
        return;
    }

    // 將輸入值轉換為數字
    const squareFeetNum = parseFloat(squareFeet);

    try {
        predictedPriceDisplay.textContent = '正在預測...';
        predictedPriceDisplay.style.color = '#ffc107'; // 黃色表示處理中

        // --- 實際 API 呼叫 ---
        // 解開註解下面的 fetch 程式碼，並替換 PREDICTION_API_URL
        /*
        const response = await fetch(PREDICTION_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ square_feet: squareFeetNum }), // 根據你的 API 需求傳送資料
        });

        if (!response.ok) {
            throw new Error(`API 請求失敗: ${response.statusText}`);
        }

        const data = await response.json();
        const predictedPrice = data.predicted_price; // 假設 API 回傳的 JSON 格式為 { predicted_price: 123456 }
        */

        // --- 使用模擬 API 進行測試 ---
        const predictedPrice = await simulatePrediction(squareFeetNum);


        // 顯示預測結果
        predictedPriceDisplay.textContent = `$${predictedPrice.toLocaleString()}`; // 格式化貨幣
        predictedPriceDisplay.style.color = '#28a745'; // 綠色表示成功

    } catch (error) {
        console.error('預測時發生錯誤:', error);
        predictedPriceDisplay.textContent = `預測失敗: ${error.message}`;
        predictedPriceDisplay.style.color = '#dc3545'; // 顯示紅色錯誤訊息
    }
});
```

**重點說明：**

1.  **`PREDICTION_API_URL`**: 你需要將 `'YOUR_PREDICTION_API_ENDPOINT'` 替換成你實際的預測 API 的網址。
2.  **`simulatePrediction` 函數**: 這個函數是為了讓你能夠在沒有實際 API 的情況下，也能測試你的儀表板。當你的 API 準備好後，記得**解除註解** `fetch` 的程式碼，並**註解掉** `simulatePrediction` 的呼叫。
3.  **`fetch` API**: 這是 JavaScript 內建的強大工具，用於發送網路請求。
    *   `method: 'POST'`：我們通常會用 POST 方法向 API 發送數據。
    *   `headers: { 'Content-Type': 'application/json' }`：告訴伺服器我們發送的數據是 JSON 格式。
    *   `body: JSON.stringify({ square_feet: squareFeetNum })`：將 JavaScript 物件轉換為 JSON 字串，並傳送給 API。請根據你的 API 文件修改傳送的參數名稱 (例如 `square_feet`)。
4.  **錯誤處理 (`try...catch`)**: 網路請求可能會出錯，所以我們使用 `try...catch` 來優雅地處理潛在的錯誤，並給使用者一個友善的提示。
5.  **結果顯示**: 將 API 回傳的預測結果 (`data.predicted_price`) 顯示在 `predictedPriceDisplay` 元素中，並進行貨幣格式化。

### 接下來該怎麼做？

1.  **儲存檔案：** 將上述 HTML、CSS (可選) 和 JavaScript 程式碼分別儲存為 `index.html`, `style.css`, 和 `script.js` 在同一個資料夾下。
2.  **替換 API URL：** 如果你已經有了預測 API，請務必將 `PREDICTION_API_URL` 替換成你的 API 端點。
3.  **啟用實際 API 呼叫：** 解開 `script.js` 中 `fetch` 的註解，並註解掉 `simulatePrediction` 的呼叫。
4.  **在瀏覽器中打開：** 用你的網頁瀏覽器打開 `index.html` 檔案。
5.  **測試！** 輸入房屋面積，點擊「預測房價」，看看預測結果！

### 總結

恭喜你！你剛剛成功地將一個預測 API 整合到了你的互動式儀表板中。這是一個非常重要的技能，它能讓你的應用程式變得更加智能和實用。你會發現，一旦掌握了 API 整合的技巧，你就能將各種不同的服務和數據源連接起來，創造出無限的可能性。

這趟旅程即將進入尾聲，但你的學習才剛剛開始！繼續探索、實驗，並享受程式帶來的樂趣吧！我們明天見！🚀