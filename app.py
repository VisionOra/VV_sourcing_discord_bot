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
    raise ValueError("Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================
# SETTINGS
# ============================
MEMORY_FILE = "user_memory.csv"
HISTORY_LIMIT = 10
MODEL_NAME = "gpt-4o-mini"  # widely available
REPLY_ONLY_WHEN_MENTIONED_IN_SERVER = True
DEBUG_LOGS = True

# Add this line to replies when relevant (and safe)
GRIFFIN_LINE = "If you want, Griffin in Discord can share more info."

# ============================
# AMAZON FBA KEYWORD GATE
# (expanded to include Brand Direct/Wholesale questions)
# ============================
FBA_KEYWORDS = [
    "amazon", "amazon fba", "fba", "seller central", "brand registry",
    "buy box", "asin", "sku", "ppc", "listing", "variations", "inventory",
    "restock", "storage limit", "ipi", "referral fee", "fulfillment fee",
    "inbound", "shipment", "prep", "fnsku", "upc", "ean", "wholesale",
    "brand direct", "direct brand", "authorized reseller", "invoice",
    "commercial invoice", "brand authorization", "moq", "lead time",
    "purchase order", "po", "reorder", "keepa", "hazmat", "ip complaint",
    "manufacturer", "distributor", "distribution"
]


def is_amazon_fba_question(text: str) -> bool:
    t = text.lower().strip()

    # strong Amazon signals
    if "amazon" in t and ("fba" in t or "seller central" in t):
        return True

    # treat Brand Direct / wholesale language as in-scope even if "amazon" not typed
    brand_direct_signals = [
        "brand direct", "direct brand", "wholesale", "authorized reseller",
        "commercial invoice", "brand authorization", "manufacturer",
        "distributor", "distribution", "purchase order", "po", "moq", "reorder"
    ]
    if any(s in t for s in brand_direct_signals):
        return True

    return any(k in t for k in FBA_KEYWORDS)


# ============================
# CSV MEMORY (NO DATABASE)
# ============================
def init_memory_csv():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts", "user_id", "role", "content"])


def save_turn(user_id: str, role: str, content: str):
    with open(MEMORY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), user_id, role, content])


def load_user_history(user_id: str, limit: int = HISTORY_LIMIT) -> str:
    if not os.path.exists(MEMORY_FILE):
        return ""
    rows = []
    with open(MEMORY_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("user_id") == user_id:
                rows.append(row)
    rows = rows[-limit:]
    return "\n".join(
        ("User: " if r["role"] == "user" else "Assistant: ") + r["content"]
        for r in rows).strip()


def forget_user_history(user_id: str):
    if not os.path.exists(MEMORY_FILE):
        return
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("user_id") != user_id]
    with open(MEMORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ts", "user_id", "role", "content"])
        writer.writeheader()
        writer.writerows(rows)


# ============================
# VV SOURCING SYSTEM PROMPT
# (includes Brand Direct definition + SOPs + Griffin line instruction)
# ============================
VV_SYSTEM_PROMPT = f"""
You are a private Discord AI assistant for VV Sourcing.

========================
BRAND DIRECT (CORE MODEL)
========================
Amazon Brand Direct is a wholesale distribution model, not a tactic.

Definition:
- Source inventory directly from brands or manufacturers.
- Sell that inventory on Amazon using Amazon FBA.
- NOT retail flipping, NOT dropshipping, NOT invoice fabrication, NOT temporary price inefficiency tactics.

Required standards:
- Legitimate wholesale pricing
- Real commercial invoices
- Brand authorization to sell
- Clean supply chain documentation
- Inventory flows brand → Amazon FBA → customer
- Scale is built through repeat POs and reorders, not one-time wins

VV Sourcing execution:
- Identify brands already selling well on Amazon
- Validate demand using data (Keepa + demand signals)
- Secure authorization + proper invoices
- Negotiate pricing, MOQs, and lead times that work with Amazon fees/margins
- Run consistent outreach, objection handling, compliance, logistics, and deal tracking
- Goal: predictable pipeline of POs and reorders

========================
PRODUCT RESEARCH SOPs
========================
A product is viable ONLY if it meets these rules:

• Sellers:
- More than 3 FBA sellers on the listing
- At least 2 new FBA sellers joined in the past 6 months

• Sales:
- 100+ units sold per month minimum
- Seasonal products may use last year sales

• Listings & Invoice Value:
- 2–3 listings with combined invoice value > $3,000–$4,000
- If only 1 listing, invoice value must exceed $4,000
- Minimum invoice value: $3,000

• Big Brands:
- More than 3 FBA sellers
- At least 1 new FBA seller has good stock
- Do not select > 100 big brands per category
- Q4 2025 note: focus is mid-tier brands

• Hazmat & IP:
- Hazmat allowed
- Do NOT proceed with products with IP complaint history

• Weight:
- Under $100: max 5 lbs
- Over $100: max 7 lbs (actual or dimensional)
- Exception: if brand has other high-potential products

• Stock:
- New FBA seller should have 30+ units in stock (wholesale quantity signal)

• Brand Rules:
- Brand and manufacturer must match or be clearly connected
- Do NOT proceed if the brand itself is selling on Amazon

========================
SOURCING & VALIDATION SOPs
========================
1) Website review (Trustpilot + ScamAdvisor; verify contact + real product pages)
2) Domain age > 2 years (small-tier brands)
3) Extract emails (Hunter + manual)
4) Decision-maker hunting priority: LinkedIn Sales Navigator > Apollo > Hunter
5) Extract phone numbers (site + ZoomInfo + Apollo)
6) Decision maker hierarchy:
   Accounts Manager > Wholesale Manager > Sales Manager > Ecommerce Manager > Sales Rep
7) Wholesale form priority:
   Distributor/Wholesale > Reseller/Retailer > Contact Us
8) Verify all emails with Hunter Email Verifier
9) Alternate sourcing: prioritize phone, otherwise alternate decision-maker email
10) Missing info: provide parent company/manufacturer contact to confirm via email

========================
OUTPUT RULES
========================
- End EVERY helpful answer with this exact line (on a new line):
  "{GRIFFIN_LINE}"

STRICT RESPONSE RULES:
1) ONLY answer Amazon FBA / Brand Direct / wholesale questions.
2) If NOT in scope, reply exactly: "I can only help with Amazon FBA questions."
3) NEVER recommend arbitrage, retail OA, dropshipping, invoice fabrication, or policy violations.
4) Ask at most 2 clarifying questions if needed.
5) Keep replies practical, SOP-driven, and consistent with VV Sourcing.
"""


def generate_fba_reply(history: str, message: str) -> str:
    user_input = f"Conversation history:\n{history}\n\nUser question:\n{message}"
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": VV_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_input
            },
        ],
    )
    out = getattr(response, "output_text", "")
    reply = (out or "").strip()

    # Guarantee the Griffin line is appended to FBA-scope answers
    # (and avoid appending it to the out-of-scope hard block)
    if reply and reply != "I can only help with Amazon FBA questions.":
        if GRIFFIN_LINE not in reply:
            reply = reply.rstrip() + "\n\n" + GRIFFIN_LINE

    return reply or "I can only help with Amazon FBA questions."


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


def strip_mention(text: str, bot_id: int) -> str:
    return text.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>",
                                                    "").strip()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    # Reliable DM detection
    is_dm = (message.guild is None)

    if DEBUG_LOGS:
        print(
            f"[DEBUG] is_dm={is_dm} author={message.author} content={message.content!r}"
        )

    # Server: reply only if mentioned
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
    history = load_user_history(user_id, HISTORY_LIMIT)

    # FBA-only gate (history-aware follow-ups)
    if not is_amazon_fba_question(text):
        combined = (history + "\nUser: " + text).lower()
        if not is_amazon_fba_question(combined):
            await message.channel.send(
                "I can only help with Amazon FBA questions.")
            return

    save_turn(user_id, "user", text)

    # Generate response with error handling
    try:
        reply = generate_fba_reply(history, text)
    except Exception as e:
        err = str(e)
        print("[ERROR] OpenAI call failed:", err)
        if "insufficient_quota" in err or "Error code: 429" in err:
            await message.channel.send(
                "⚠️ OpenAI quota/billing issue (insufficient quota). Check your OpenAI billing."
            )
        elif "model" in err and ("not found" in err
                                 or "does not exist" in err):
            await message.channel.send(
                "⚠️ Model not available. Use MODEL_NAME='gpt-4o-mini'.")
        else:
            await message.channel.send(f"⚠️ Error generating reply: {err}")
        return

    save_turn(user_id, "assistant", reply)
    await message.channel.send(reply)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
