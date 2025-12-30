import os
import csv
import time
import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

# ============================
# ENV
# ============================
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DISCORD_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================
# SETTINGS
# ============================
MEMORY_FILE = "user_memory.csv"
HISTORY_LIMIT = 10
MODEL_NAME = "gpt-4o-mini"  # safest + widely enabled
REPLY_ONLY_WHEN_MENTIONED_IN_SERVER = True
DEBUG_LOGS = True

# ============================
# AMAZON FBA KEYWORD GATE
# ============================
FBA_KEYWORDS = [
    "amazon fba", "fba", "seller central", "brand registry", "buy box", "asin",
    "sku", "ppc", "listing", "inventory", "restock", "storage limit", "ipi",
    "referral fee", "fulfillment fee", "shipment", "prep", "fnsku",
    "wholesale", "brand direct", "authorized reseller", "invoice", "moq",
    "lead time", "reorder", "keepa", "hazmat", "ip complaint", "manufacturer"
]


def is_amazon_fba_question(text: str) -> bool:
    t = text.lower()
    if "amazon" in t and ("fba" in t or "seller central" in t):
        return True
    return any(k in t for k in FBA_KEYWORDS)


# ============================
# CSV MEMORY
# ============================
def init_memory_csv():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts", "user_id", "role", "content"])


def save_turn(user_id, role, content):
    with open(MEMORY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), user_id, role, content])


def load_user_history(user_id, limit=HISTORY_LIMIT):
    if not os.path.exists(MEMORY_FILE):
        return ""
    rows = []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["user_id"] == user_id:
                rows.append(row)
    rows = rows[-limit:]
    return "\n".join(
        ("User: " if r["role"] == "user" else "Assistant: ") + r["content"]
        for r in rows)


def forget_user_history(user_id):
    if not os.path.exists(MEMORY_FILE):
        return
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["user_id"] != user_id]
    with open(MEMORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ts", "user_id", "role", "content"])
        writer.writeheader()
        writer.writerows(rows)


# ============================
# VV SOURCING SYSTEM PROMPT (WITH SOPs)
# ============================
VV_SYSTEM_PROMPT = """
You are a private Discord AI assistant for VV Sourcing.

================================
CORE MODEL: AMAZON BRAND DIRECT
================================
Amazon Brand Direct is a wholesale distribution model, not a tactic.

- Inventory is sourced directly from brands or manufacturers
- Inventory is placed into Amazon FBA
- This is NOT retail arbitrage, NOT dropshipping, NOT fabricated invoices
- You must have legitimate wholesale pricing
- You must have real commercial invoices
- You must have brand authorization to sell
- Clean supply chain documentation is mandatory
- Inventory flows brand → Amazon FBA → customer
- Scale happens through reorders, not one-time wins

================================
PRODUCT RESEARCH SOPs
================================
A product is considered viable ONLY if it meets these rules:

• Sellers:
- More than 3 FBA sellers
- At least 2 new FBA sellers joined in the last 6 months

• Sales:
- Minimum 100+ units sold per month
- Seasonal products may use last year's sales data

• Listings & Invoice Value:
- 2–3 listings with combined invoice value $3,000–$4,000+
- If only 1 listing → invoice must exceed $4,000
- Minimum invoice value: $3,000

• Big Brands:
- More than 3 FBA sellers
- At least 1 new FBA seller with strong stock
- Do not select more than 100 big brands per category
- Q4 2025 exception: focus is mid-tier brands

• Hazmat & IP:
- Hazmat is allowed
- Products with IP complaint history are NOT allowed

• Weight:
- Under $100 → max 5 lbs
- Over $100 → max 7 lbs (actual or dimensional)
- Exceptions allowed if brand has other strong SKUs

• Stock:
- New FBA seller must have 30+ units in stock

• Brand Rules:
- Brand and manufacturer must match or be clearly connected
- Brand CANNOT be selling on Amazon themselves

================================
SOURCING & VALIDATION SOPs
================================
Follow these steps IN ORDER:

1) Website Review:
- Check Trustpilot & ScamAdvisor
- Verify contact info and real product pages

2) Domain Age:
- Must be older than 2 years (for small-tier brands)

3) Email Extraction:
- Website + Email Hunter

4) Decision Maker Priority:
Accounts Manager > Wholesale Manager > Sales Manager > Ecommerce Manager > Sales Rep

5) Tools for Contact Discovery:
- LinkedIn Sales Navigator
- Apollo.io
- Hunter.io

6) Phone Numbers:
- Website, ZoomInfo, Apollo.io

7) Wholesale Form Priority:
Distributor/Wholesale Form > Reseller Form > Contact Us

8) Email Verification:
- All emails verified with Hunter.io

9) Alternate Contacts:
- Provide phone first, then alternate decision-maker email

10) Missing Info:
- Contact parent company or manufacturer

================================
STRICT RESPONSE RULES
================================
1) ONLY answer Amazon FBA questions
2) ONLY recommend Brand Direct / wholesale strategies
3) NEVER recommend arbitrage, dropshipping, or policy violations
4) If NOT Amazon FBA related → reply exactly:
   "I can only help with Amazon FBA questions."
5) Ask max 2 clarifying questions if needed
6) Speak confidently as VV Sourcing
"""


def generate_fba_reply(history, message):
    response = client.responses.create(
        model=MODEL_NAME,
        input=[{
            "role": "system",
            "content": VV_SYSTEM_PROMPT
        }, {
            "role":
            "user",
            "content":
            f"Conversation history:\n{history}\n\nUser question:\n{message}"
        }])
    return (getattr(response, "output_text", "") or "").strip() or \
           "I can only help with Amazon FBA questions."


# ============================
# DISCORD BOT
# ============================
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    init_memory_csv()
    print(f"✅ Logged in as {bot.user}")


@bot.command(name="forget")
async def forget(ctx):
    forget_user_history(str(ctx.author.id))
    await ctx.send("🗑️ Your chat history has been deleted.")


def strip_mention(text, bot_id):
    return text.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>",
                                                    "").strip()


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    is_dm = (message.guild is None)

    if DEBUG_LOGS:
        print(f"[DEBUG] DM={is_dm} | {message.author}: {message.content}")

    if not is_dm and REPLY_ONLY_WHEN_MENTIONED_IN_SERVER:
        if bot.user not in message.mentions:
            return

    text = message.content.strip()
    if not is_dm:
        text = strip_mention(text, bot.user.id)

    if not text:
        await message.channel.send("Mention me with your Amazon FBA question.")
        return

    user_id = str(message.author.id)
    history = load_user_history(user_id)

    if not is_amazon_fba_question(text):
        combined = (history + text).lower()
        if not is_amazon_fba_question(combined):
            await message.channel.send(
                "I can only help with Amazon FBA questions.")
            return

    save_turn(user_id, "user", text)

    try:
        reply = generate_fba_reply(history, text)
    except Exception as e:
        await message.channel.send(
            "⚠️ Error generating response. Check OpenAI billing or model access."
        )
        print("[ERROR]", e)
        return

    save_turn(user_id, "assistant", reply)
    await message.channel.send(reply)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
