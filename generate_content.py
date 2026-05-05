import os
import json
import time
import google.generativeai as genai

# 1. 設定 Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("找不到 GEMINI_API_KEY 環境變數")

genai.configure(api_key=api_key)

# 準備好兩個版本的模型
model_primary = genai.GenerativeModel('gemini-2.5-flash')
model_fallback = genai.GenerativeModel('gemini-1.5-flash')

# ==========================================
# 🌟 建立一個有「備援機制 (Fallback)」的智慧生成函式
# ==========================================
def safe_generate(prompt):
    try:
        print("➡️ 嘗試呼叫 gemini-2.5-flash 模型...")
        return model_primary.generate_content(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        # 如果捕捉到「額度不足 (quota)」或「頻率限制 (429/exhausted)」的錯誤
        if "quota" in error_msg or "429" in error_msg or "exhausted" in error_msg:
            print("⚠️ 2.5 版本額度耗盡，啟動備援：自動切換至 gemini-1.5-flash 繼續執行！")
            return model_fallback.generate_content(prompt)
        else:
            # 如果是其他未知的錯誤，就直接報錯
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

if not target_day:
    print("🎉 所有課程都已經生成完畢了！")
    exit(0)

day_num = target_day["day"]
day_title = target_day["title"]
print(f"\n🚀 開始生成 Day {day_num}: {day_title} 的內容...")

# 4. 準備資料夾路徑
folder_path = f"content/{course_id}/day-{day_num}"
os.makedirs(folder_path, exist_ok=True)

# 5. 呼叫 AI 生成 Markdown 教學文章
prompt_lesson = f"""
你是一位專業的程式導師。請幫我撰寫一篇給程式初學者的教材。
主題：【第 {day_num} 天：{day_title}】
要求：
1. 長度約 600-800 字。
2. 語氣要鼓勵、輕鬆。
3. 必須包含具體的 Python 程式碼範例。
4. 使用 Markdown 格式。
"""
print("\n正在生成教材文章...")
# 這裡改用我們自己寫的 safe_generate 函式
response_lesson = safe_generate(prompt_lesson) 
with open(f"{folder_path}/lesson.md", "w", encoding="utf-8") as f:
    f.write(response_lesson.text)

# ==========================================
# 🛑 暫停 30 秒，避免觸發每分鐘頻率限制
print("\n等待 30 秒讓 AI 喘口氣...")
time.sleep(30)
# ==========================================

# 6. 呼叫 AI 生成 JSON 格式的測驗題
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
# 這裡也改用我們自己寫的 safe_generate 函式
response_quiz = safe_generate(prompt_quiz)

# 清理 AI 回傳可能夾帶的 Markdown 標籤
raw_json = response_quiz.text.strip()
if raw_json.startswith("```json"):
    raw_json = raw_json[7:]
if raw_json.endswith("```"):
    raw_json = raw_json[:-3]

with open(f"{folder_path}/quiz.json", "w", encoding="utf-8") as f:
    f.write(raw_json.strip())

# 7. 更新 index.json，把該天標記為已完成
target_day["generated"] = True
target_day["path"] = folder_path

with open("index.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Day {day_num} 內容生成完畢並已更新目錄！")
