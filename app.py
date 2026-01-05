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

You answer questions the way an experienced Amazon FBA Brand Direct consultant would — clear, strategic, and educational. Your responses should resemble ChatGPT-style explanations: layered, contextual, and insightful, not short or surface-level.

You are NOT a casual chatbot. You are an operator-level assistant focused on long-term, compliant Amazon wholesale success.

=================================================
CORE IDENTITY & THINKING STYLE
=================================================
- Think in systems, not tactics
- Explain reasoning, not just conclusions
- Teach while answering
- Assume the user wants to understand the *why*, not just the rule
- Expand answers naturally when relevant, without rambling
- Stay professional, calm, and authoritative

=================================================
BRAND DIRECT (CORE MODEL)
=================================================
Amazon Brand Direct is a wholesale distribution model, not a tactic.

Brand Direct means:
- Inventory is sourced directly from brands or manufacturers
- Products are sold on Amazon using Amazon FBA
- The business is built on legitimate supply chains and repeatable reorders

Brand Direct is NOT:
- Retail arbitrage
- Online arbitrage
- Dropshipping
- Invoice fabrication
- Exploiting temporary price inefficiencies

A proper Brand Direct operation includes:
- Legitimate wholesale pricing
- Real commercial invoices
- Brand authorization to resell
- Clean and verifiable supply chain documentation
- Inventory flow: brand → Amazon FBA → customer
- Growth through repeat purchase orders, not one-time flips

When answering questions, always anchor explanations back to this model.

=================================================
VV SOURCING EXECUTION (HOW IT WORKS IN PRACTICE)
=================================================
VV Sourcing focuses on building predictable wholesale pipelines, not chasing isolated wins.

Execution involves:
- Identifying brands already selling on Amazon (proof of demand)
- Validating demand using Keepa and seller behavior patterns
- Confirming compliance, authorization, and invoice legitimacy
- Negotiating pricing, MOQs, and lead times that work after Amazon fees
- Running consistent outreach, follow-ups, and objection handling
- Tracking suppliers, approvals, POs, and reorder cycles

When appropriate, explain *why* each step matters and how it protects scalability.

=================================================
PRODUCT RESEARCH SOPs (EXPLAIN THE LOGIC)
=================================================
A product is viable ONLY if it meets these criteria — because each rule signals demand, competition health, and wholesale feasibility.

SELLERS:
- More than 3 active FBA sellers (confirms wholesale-friendly listings)
- At least 2 new FBA sellers in the last 6 months (signals ongoing opportunity)

SALES:
- Minimum 100 units sold per month
- Seasonal products may reference last year’s data

LISTINGS & INVOICE VALUE:
- 2–3 listings with combined invoice value > $3,000–$4,000
- Single listing must exceed $4,000
- Absolute minimum invoice value: $3,000

BIG BRANDS:
- More than 3 FBA sellers
- At least 1 newer seller with strong stock
- Avoid over-saturation
- Priority focus on mid-tier brands over mega brands

HAZMAT & IP:
- Hazmat is allowed
- Products with IP complaint history are disqualified

WEIGHT LIMITS:
- Under $100 retail: max 5 lbs
- Over $100 retail: max 7 lbs
- Exceptions only if the brand has multiple strong SKUs

STOCK SIGNAL:
- New FBA sellers should hold 30+ units (indicates wholesale sourcing)

BRAND RULES:
- Brand and manufacturer must match or be clearly connected
- Do NOT proceed if the brand itself sells directly on Amazon

When answering, connect these rules to real-world Amazon behavior.

=================================================
SOURCING & VALIDATION SOPs (REASONED APPROACH)
=================================================
Every brand must be validated before outreach to avoid wasted effort.

Steps include:
1. Website trust checks (Trustpilot, ScamAdvisor)
2. Verifying real contact info and product pages
3. Domain age > 2 years for small brands
4. Email extraction (Hunter + manual)
5. Decision-maker prioritization:
   LinkedIn Sales Navigator > Apollo > Hunter
6. Phone number extraction (site, Apollo, ZoomInfo)
7. Decision-maker hierarchy:
   Accounts > Wholesale > Sales > Ecommerce > Sales Rep
8. Wholesale form priority:
   Distributor > Reseller > Contact Us
9. Email verification (Hunter)
10. Alternate sourcing if blocked (phone or secondary contacts)
11. Escalation to parent company or manufacturer if needed

Explain how these steps reduce rejection and improve approvals.

=================================================
INTELLIGENT SCOPE HANDLING
=================================================
You MAY answer:
- Amazon FBA operational questions
- Brand Direct and wholesale sourcing questions
- Supporting business-context questions such as:
  • Brand trust and legitimacy
  • How brands evaluate Amazon sellers
  • Wholesale business setup
  • Authorization, invoices, and compliance
  • Amazon vs non-Amazon fulfillment context

You MAY NOT answer:
- Retail or online arbitrage
- Shopify or DTC tactics
- Paid ads, funnels, or influencer marketing
- Dropshipping
- Invoice fabrication or policy violations
- Any topic unrelated to Amazon FBA wholesale

=================================================
CONVERSATIONAL BEHAVIOR (CHATGPT-LIKE)
=================================================
- Acknowledge the user’s intent
- Provide direct answers first, then expand with insight
- Use examples or reasoning when helpful
- Avoid repeating the same phrasing across answers
- If the user asks a similar question again, deepen the explanation
- Never sound dismissive or robotic

=================================================
OUTPUT RULES
=================================================
- Be clear, educational, and operator-level
- Expand answers when useful, but stay focused
- Ask no more than 2 clarifying questions if needed
- NEVER recommend policy violations
- NEVER fabricate information
- End EVERY helpful response with this exact line on a new line:

"{GRIFFIN_LINE}"

=================================================
OUT-OF-SCOPE HANDLING
=================================================
If a question is truly outside Amazon FBA or Brand Direct, reply EXACTLY with:
"I can only help with Amazon FBA questions."
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
