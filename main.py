import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# گرفتن توکن‌ها از Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# API URL مدل Hugging Face برای تحلیل احساسات (انگلیسی)
API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# تابع ارسال درخواست به Hugging Face
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# شروع بات
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Send me a sentence and I will analyze its sentiment.")

# پردازش پیام کاربر
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    await update.message.reply_text("Analyzing...")
    output = query({"inputs": user_text})
    
    if isinstance(output, list):
        label = output[0][0]["label"]
        score = output[0][0]["score"]
        await update.message.reply_text(f"Sentiment: {label}\nConfidence: {score:.2f}")
    else:
        await update.message.reply_text("Error: Could not analyze the text.")

# اجرای بات
app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("Bot is running...")
app.run_polling()
