Building a Telegram bot that receives the text of Twitter users' tweets, then analyzes them with the BERT and SVM models, and sometimes may be able to show

# 🤖 SENTIMENT_AI_BOT

ربات تلگرام برای تحلیل احساسات متن و توییت به سه دسته **منفی**، **خنثی** و **مثبت**.

---

## ✨ قابلیت‌ها

- تحلیل احساسات متن به سه کلاس: 😡 منفی | 😐 خنثی | 😊 مثبت
- نمایش نمودار احتمال هر احساس
- پشتیبانی از متن‌های انگلیسی و توییت
- سرعت بالا روی CPU

---

## 🧠 مدل

| مشخصه | مقدار |
|--------|-------|
| مدل | `twitter-roberta-base-sentiment` |
| منبع | Cardiff NLP — HuggingFace |
| آموزش | ۱۲۴ میلیون توییت واقعی |
| نیاز به GPU | خیر |

---

## 📁 ساختار فایل‌ها

```
├── bot/
│   └── telegram_bot.py        ← کد اصلی ربات
├── .github/
│   └── workflows/
│       └── run_bot.yml        ← اجرای خودکار (GitHub Actions)
├── requirements.txt           ← وابستگی‌ها
└── README.md
```

---

## 🚀 راه‌اندازی

**۱. Secret تنظیم کنید:**

در GitHub: `Settings → Secrets → Actions → New secret`

| نام | توضیح |
|-----|-------|
| `TELEGRAM_TOKEN` | توکن ربات از @BotFather |

**۲. اجرا کنید:**

تب `Actions` → `Run Sentiment Telegram Bot` → `Run workflow`

---

## 💬 دستورات ربات

| دستور | توضیح |
|-------|-------|
| `/start` | شروع و راهنما |
| `/about` | اطلاعات مدل |
| `/demo` | نمونه آنالیز |
| ارسال متن | تحلیل احساسات |

---

## 📬 ربات تلگرام

[@SENTIMENT_AGENT_BOT](https://t.me/SENTIMENT_AGENT_BOT)
