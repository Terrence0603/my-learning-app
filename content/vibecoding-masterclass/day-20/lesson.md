好的，程式探險家們！準備好讓你的程式更上一層樓了嗎？

---

## 【第 20 天：自動化排程 API 資料更新】

嘿，程式探險家們！恭喜你，我們已經一起走過 19 天的旅程了！還記得我們在之前的課程中學習如何呼叫 API 取得資料嗎？那是你第一次與網路世界對話，感覺很酷，對吧？

但是，想像一下：如果你的老闆說「我每天早上 9 點都需要最新的天氣資料」或者「我們的報表每小時都要更新股市行情」... 你難道要每天準時坐在電腦前，手動執行你的 Python 程式碼，一次又一次地取得資料嗎？

那也太累了吧！身為一個聰明的程式設計師，我們當然要讓電腦幫我們完成這些重複性高又耗時的工作！今天，我們就要賦予你的程式「生命」，讓它自己動起來，在我們設定的時間自動更新 API 資料！

準備好了嗎？讓我們開始吧！

### 為什麼要自動化排程？

1.  **節省時間與精力**：把重複性任務交給電腦，你可以專注於更有創意、更具挑戰性的工作。
2.  **確保資料新鮮度**：自動定時更新，保證你總是拿到最新、最即時的資訊。
3.  **減少人為錯誤**：手動操作容易出錯，程式碼執行則精準無誤。
4.  **解放你的雙手**：讓程式在背景默默工作，你甚至可以去喝杯咖啡、散個步，回來資料就更新好了！

這就像是為你的程式設定了一個鬧鐘和一個執行列表，它會乖乖地在時間到了的時候，自動完成你交代的所有任務。

### 我們會使用哪些工具？

今天，我們要介紹一個超級好用的 Python 函式庫，叫做 `schedule`。它能讓你在 Python 程式碼中，非常直觀地設定各種排程任務，像是「每 10 秒執行一次」、「每天早上 8 點執行」等等。

當然，`requests` 函式庫（用於發送 API 請求）和 `time` 函式庫（用於讓程式稍微休息一下）也依然是我們的老朋友。

#### **第一步：安裝 `schedule` 函式庫**

如果你的環境中還沒有 `schedule`，請打開你的終端機或命令提示字元，輸入以下指令：

```bash
pip install schedule
```

### 實際操作：讓程式自動更新資料！

想像一下，我們要定期去一個假想的 API 獲取一些待辦事項（Todos）的資料。

#### **程式碼分解**

1.  **取得 API 資料的函數**：首先，我們需要一個函數來封裝取得 API 資料的邏輯。

    ```python
    import requests
    import json # 通常用於處理 JSON 資料，雖然 requests.json() 已處理，但有助理解

    def fetch_and_process_data():
        """
        這個函數負責呼叫 API 並處理回傳的資料。
        """
        print("--- 正在自動取得最新資料... ---")
        api_url = "https://jsonplaceholder.typicode.com/todos/1" # 一個免費的測試 API
        try:
            response = requests.get(api_url)
            response.raise_for_status() # 檢查 HTTP 狀態碼，如果不是 200 會拋出異常
            data = response.json()
            print(f"✅ 成功取得資料：{data['title']}")
            # 你可以在這裡進一步處理資料，例如：
            # - 將資料存入檔案 (with open('data.json', 'w') as f: json.dump(data, f))
            # - 將資料存入資料庫
            # - 發送通知等等
        except requests.exceptions.RequestException as e:
            print(f"❌ 取得資料時發生錯誤：{e}")
        print("--- 資料更新完成。 ---")
    ```
    這個 `fetch_and_process_data` 函數會去呼叫一個測試用的 API，並印出取得的資料標題。當你實際應用時，你可以修改 `api_url` 並在 `print(f"✅ 成功取得資料...")` 之後加入你真正想對資料進行的處理，例如寫入資料庫或生成報表。

2.  **設定排程並啟動**：現在，我們要使用 `schedule` 函式庫來告訴 Python 什麼時候執行 `fetch_and_process_data`。

    ```python
    import schedule
    import time

    # ... (接續上面的 fetch_and_process_data 函數定義) ...

    # 設定排程任務
    print("🚀 設定排程：每隔 5 秒鐘執行一次資料更新。")
    # 你可以這樣設定不同的排程：
    # schedule.every(10).seconds.do(fetch_and_process_data)  # 每 10 秒
    # schedule.every(1).minutes.do(fetch_and_process_data)   # 每 1 分鐘
    # schedule.every().hour.do(fetch_and_process_data)       # 每小時
    # schedule.every().day.at("10:30").do(fetch_and_process_data) # 每天 10:30
    # schedule.every().monday.do(fetch_and_process_data)     # 每週一
    # schedule.every().wednesday.at("13:15").do(fetch_and_process_data) # 每週三 13:15

    schedule.every(5).seconds.do(fetch_and_process_data)

    # 啟動排程器，讓它持續運行
    print("🟢 排程器已啟動，等待執行任務... (按下 Ctrl+C 停止程式)")
    while True:
        schedule.run_pending() # 檢查是否有任務待執行，並執行它們
        time.sleep(1)          # 讓程式暫停 1 秒，避免過度佔用 CPU
    ```

#### **完整的程式碼範例**

將上面兩部分的程式碼組合起來，存為 `auto_update.py` 檔案：

```python
import requests
import json
import schedule
import time

def fetch_and_process_data():
    """
    這個函數負責呼叫 API 並處理回傳的資料。
    """
    print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} - 正在自動取得最新資料... ---")
    api_url = "https://jsonplaceholder.typicode.com/todos/1" # 一個免費的測試 API
    try:
        response = requests.get(api_url)
        response.raise_for_status() # 檢查 HTTP 狀態碼，如果不是 200 會拋出異常
        data = response.json()
        print(f"✅ 成功取得資料：{data['title']}")
        # 這裡可以加入你實際的資料處理邏輯，例如：
        # with open('latest_todo.json', 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)
        # print("資料已儲存到 latest_todo.json")
    except requests.exceptions.RequestException as e:
        print(f"❌ 取得資料時發生錯誤：{e}")
    print("--- 資料更新完成。 ---")

# --- 設定排程 ---
print("🚀 設定排程：每隔 5 秒鐘執行一次資料更新。")
schedule.every(5).seconds.do(fetch_and_process_data)

# --- 啟動排程器 ---
print("🟢 排程器已啟動，等待執行任務... (按下 Ctrl+C 停止程式)")
while True:
    schedule.run_pending() # 檢查是否有任務待執行，並執行它們
    time.sleep(1)          # 讓程式暫停 1 秒，避免過度佔用 CPU
```

#### **如何執行？**

1.  將上述程式碼儲存為 `auto_update.py`。
2.  打開你的終端機或命令提示字元，切換到該檔案所在的目錄。
3.  執行指令：`python auto_update.py`

你會看到程式碼開始運行，然後每隔 5 秒鐘，它就會自動印出從 API 取得的資料！是不是很神奇？要停止程式，只需在終端機中按下 `Ctrl + C`。

### 恭喜你！

現在，你已經從一個單純的資料索取者，晉升為一名**自動化資料管理師**！這個技能在實際工作中非常有用，無論是數據分析、網站監控還是自動報告，自動化排程都能幫你大大提升效率。

**挑戰與思考：**

*   你能嘗試將取得的資料儲存到一個文字檔或 JSON 檔中嗎？
*   嘗試設定不同的排程時間，例如每分鐘更新一次。
*   如果 API 呼叫失敗，你能設計一個錯誤通知機制，例如發送電子郵件或記錄到日誌檔嗎？

**恭喜你，完成了第 20 天的學習！** 你正在一步步地掌握程式設計的奧秘，讓程式為你服務！休息一下，明天我們將探索更多有趣的內容！