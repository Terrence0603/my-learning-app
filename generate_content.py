import os
import json
import time
import google.generativeai as genai

# 1. 設定 Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("找不到 GEMINI_API_KEY 環境變數")

genai.configure(api_key=api_key)

model_primary = genai.GenerativeModel('gemini-2.5-flash')
model_fallback = genai.GenerativeModel('gemini-2.5-flash-lite')

def safe_generate(prompt):
    try:
        print("➡️ 嘗試呼叫 gemini-2.5-flash 模型...")
        return model_primary.generate_content(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg or "exhausted" in error_msg:
            print("⚠️ 2.5 版本額度耗盡，啟動備援：自動切換至 gemini-2.5-flash-lite 繼續執行！")
            return model_fallback.generate_content(prompt)
        else:
            raise e

# 2. 讀取目前的目錄狀態
with open("index.json", "r", encoding="utf-8") as f:
    data = json.load(f)

course = data["courses"][0]
course_id = course["id"]

# 3. 尋找下一個尚未生成的課程
target_day = None
for day in course["days"]:
    if not day.get("generated", False):
        target_day = day
        break

# 🔥 黑科技：如果課程都生成完了，讓 AI 自動擴展新的一天！
if not target_day:
    print("💡 目前大綱已全部生成完畢，正在請 AI 規劃新的一天...")
    last_day_num = course["days"][-1]["day"]
    next_day_num = last_day_num + 1

    # 整理歷史大綱給 AI 參考 (取最後 5 天就好)
    history_titles = [f"Day {d['day']}: {d['title']}" for d in course["days"][-5:]]
    history_text = "\n".join(history_titles)

    prompt_new_day = f"""
    你是一位程式課程規劃師。目前這個「Vibecoding x 專案迭代 x Python」課程已經進行到 Day {last_day_num}。
    最近幾天的課程標題是：
    {history_text}

    請依照學習邏輯，幫我規劃【Day {next_day_num}】的進階課程標題。
    你「只能」回傳一個純文字標題，不要任何多餘的引言或 Markdown 符號。例如：「了解 API 與 JSON 格式」或「實戰：用 Python 寫一個簡單的爬蟲」。
    """
    
    response_new_title = safe_generate(prompt_new_day)
    new_title = response_new_title.text.strip()
    
    # 建立新的一天
    target_day = {
        "day": next_day_num,
        "title": new_title,
        "generated": False
    }
    course["days"].append(target_day)
    print(f"🆕 成功擴增課程大綱：Day {next_day_num} - {new_title}")

    # 先存檔一次目錄，以防後續出錯
    with open("index.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

day_num = target_day["day"]
day_title = target_day["title"]
print(f"\n🚀 開始生成 Day {day_num}: {day_title} 的內容...")

# 4. 準備資料夾路徑
folder_path = f"content/{course_id}/day-{day_num}"
os.makedirs(folder_path, exist_ok=True)

# 5. 呼叫 AI 生成教學文章
prompt_lesson = f"""
你是一位專業的程式導師。請幫我撰寫一篇給初學者的教材。
主題：【第 {day_num} 天：{day_title}】
要求：
1. 長度約 600-800 字。
2. 語氣要鼓勵、輕鬆。
3. 必須包含具體的程式碼範例。
4. 使用 Markdown 格式。
"""
print("\n正在生成教材文章...")
response_lesson = safe_generate(prompt_lesson) 
with open(f"{folder_path}/lesson.md", "w", encoding="utf-8") as f:
    f.write(response_lesson.text)

print("\n等待 30 秒讓 AI 喘口氣...")
time.sleep(30)

# 6. 呼叫 AI 生成測驗題
prompt_quiz = f"""
根據你剛才撰寫的【{day_title}】教材，出一份包含 3 題單選題的測驗。
你「只能」回傳純 JSON 格式的字串，不要包含任何 Markdown 標記 (如 ```json) 或其他文字。
JSON 格式必須完全符合以下結構：
{{
  "questions": [
    {{
      "id": "q1",
      "type": "single_choice",
      "question": "題目內容",
      "options": ["選項1", "選項2", "選項3", "選項4"],
      "answerIndex": 0,
      "explanation": "為什麼這題答案是這個的解釋"
    }}
  ]
}}
"""
print("\n正在生成測驗題...")
response_quiz = safe_generate(prompt_quiz)

raw_json = response_quiz.text.strip()
if raw_json.startswith("```json"):
    raw_json = raw_json[7:]
if raw_json.endswith("```"):
    raw_json = raw_json[:-3]

with open(f"{folder_path}/quiz.json", "w", encoding="utf-8") as f:
    f.write(raw_json.strip())

# 7. 更新狀態並存檔
target_day["generated"] = True
target_day["path"] = folder_path

with open("index.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Day {day_num} 內容生成完畢並已更新目錄！")
