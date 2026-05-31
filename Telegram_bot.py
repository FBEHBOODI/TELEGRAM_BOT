"""
🤖 Telegram Sentiment Bot — بدون نیاز به Colab
=================================================
از مدل آماده HuggingFace استفاده می‌کند:
  cardiffnlp/twitter-roberta-base-sentiment-latest
  (آموزش‌دیده روی توییت‌های واقعی — رایگان و بدون توکن)

فقط یک Secret در GitHub لازم است:
  TELEGRAM_TOKEN  ← از @BotFather
"""

import os, io, re, time, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import pipeline
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
MODEL_NAME     = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# برچسب‌های مدل → فارسی
LABEL_FA = {
    'negative': '😡 منفی',
    'neutral':  '😐 خنثی',
    'positive': '😊 مثبت',
}
BAR_COLORS = ['#e74c3c', '#95a5a6', '#2ecc71']


# ══════════════════════════════════════════════════════
# لود مدل — فقط یک بار در startup
# ══════════════════════════════════════════════════════
log.info(f"⏳ دانلود و لود مدل از HuggingFace: {MODEL_NAME}")
classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    top_k=None,          # همه ۳ کلاس را برمی‌گرداند
    device=-1            # CPU — سبک و بدون GPU
)
log.info("✅ مدل آماده است")


# ══════════════════════════════════════════════════════
# توابع پردازش
# ══════════════════════════════════════════════════════
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r'http\S+|www\S+', '', t)
    t = re.sub(r'@\w+', '@user', t)      # توییتر استایل
    t = re.sub(r'#(\w+)', r'\1', t)
    return t.strip()


def predict(text: str) -> dict:
    results = classifier(clean_text(text), truncation=True, max_length=128)[0]
    # مرتب‌سازی: negative, neutral, positive
    scores = {r['label'].lower(): round(r['score'] * 100, 1) for r in results}
    neg = scores.get('negative', 0)
    neu = scores.get('neutral',  0)
    pos = scores.get('positive', 0)
    pred_label = max(scores, key=scores.get)
    return {
        'label_fa': LABEL_FA[pred_label],
        'label_en': pred_label,
        'conf': scores[pred_label],
        'neg': neg, 'neu': neu, 'pos': pos,
    }


def make_chart(r: dict) -> io.BytesIO:
    cats   = ['😡 Neg', '😐 Neu', '😊 Pos']
    values = [r['neg'], r['neu'], r['pos']]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(cats, values, color=BAR_COLORS, alpha=0.85,
                  edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{val:.1f}%", ha='center', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 118)
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Sentiment Analysis (RoBERTa)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0); plt.close()
    return buf


# ══════════════════════════════════════════════════════
# Telegram Handlers
# ══════════════════════════════════════════════════════
async def cmd_start(update: Update, _):
    await update.message.reply_text(
        "👋 سلام! به ربات تحلیل احساسات خوش آمدید.\n\n"
        "🤖 مدل: RoBERTa (آموزش‌دیده روی توییت)\n"
        "📊 کلاس‌ها: منفی | خنثی | مثبت\n\n"
        "📝 هر متن یا توییتی بفرستید تا آنالیز شود.\n\n"
        "/demo — نمونه آنالیز\n"
        "/about — اطلاعات مدل"
    )


async def cmd_about(update: Update, _):
    await update.message.reply_text(
        "ℹ️ *اطلاعات مدل*\n\n"
        "🔵 *مدل:* `twitter-roberta-base-sentiment`\n"
        "🏠 *منبع:* HuggingFace (Cardiff NLP)\n"
        "📚 *آموزش:* روی ۱۲۴ میلیون توییت\n"
        "⚙️ *Device:* CPU\n"
        "🆓 *توکن API:* نیاز ندارد",
        parse_mode='Markdown'
    )


async def cmd_demo(update: Update, _):
    kb = [[
        InlineKeyboardButton("😊 مثبت", callback_data="demo_pos"),
        InlineKeyboardButton("😡 منفی", callback_data="demo_neg"),
        InlineKeyboardButton("😐 خنثی", callback_data="demo_neu"),
    ]]
    await update.message.reply_text(
        "یک نمونه انتخاب کنید:",
        reply_markup=InlineKeyboardMarkup(kb)
    )


DEMO_TEXTS = {
    'demo_pos': "I absolutely love this airline! Great service and on time.",
    'demo_neg': "Worst flight ever. Delayed 4 hours and lost my luggage. Never again.",
    'demo_neu': "The flight departed from Terminal 2 at 10am and landed on schedule.",
}


async def on_button(update: Update, _):
    q = update.callback_query
    await q.answer()
    text = DEMO_TEXTS.get(q.data)
    if text:
        await q.message.reply_text(f"📝 نمونه:\n`{text}`", parse_mode='Markdown')
        await do_analysis(q.message, text)


async def on_text(update: Update, _):
    await do_analysis(update.message, update.message.text.strip())


async def do_analysis(msg, text: str):
    if not text:
        return
    await msg.reply_text("⏳ در حال آنالیز...")

    t0 = time.time()
    r  = predict(text)
    ms = (time.time() - t0) * 1000

    reply = (
        f"📊 *نتیجه تحلیل احساسات*\n\n"
        f"📝 `{text[:100]}{'...' if len(text)>100 else ''}`\n\n"
        f"🎯 *نتیجه:* {r['label_fa']}\n"
        f"💯 *اطمینان:* {r['conf']}%\n"
        f"⏱ *زمان:* {ms:.0f}ms\n\n"
        f"📈 منفی: {r['neg']}% | خنثی: {r['neu']}% | مثبت: {r['pos']}%"
    )
    await msg.reply_text(reply, parse_mode='Markdown')
    await msg.reply_photo(
        photo=make_chart(r),
        caption="📊 نمودار احتمال هر احساس"
    )


# ══════════════════════════════════════════════════════
# اجرا
# ══════════════════════════════════════════════════════
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('about', cmd_about))
    app.add_handler(CommandHandler('demo',  cmd_demo))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("🚀 Bot is running...")
    app.run_polling(drop_pending_updates=True)


if __name__ == '__main__':
    main()