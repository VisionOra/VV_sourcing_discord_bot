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
MODEL_NAME = "gpt-4o-mini"
REPLY_ONLY_WHEN_MENTIONED_IN_SERVER = True
DEBUG_LOGS = True

GRIFFIN_LINE = "If you want, Griffin in Discord can share more info."

DISCORD_CHAR_LIMIT = 2000
SAFE_SEND_LIMIT = 1900

# ============================
# AMAZON FBA KEYWORD GATE
# ============================
FBA_KEYWORDS = [
    "amazon", "amazon fba", "fba", "seller central", "brand registry",
    "buy box", "asin", "sku", "ppc", "listing", "variations", "inventory",
    "restock", "storage limit", "ipi", "referral fee", "fulfillment fee",
    "inbound", "shipment", "prep", "fnsku", "upc", "ean", "wholesale",
    "brand direct", "direct brand", "authorized reseller", "invoice",
    "commercial invoice", "brand authorization", "moq", "lead time",
    "purchase order", "po", "reorder", "keepa", "hazmat", "ip complaint",
    "manufacturer", "distributor", "distribution", "map", "msrp"
]


def is_amazon_fba_question(text: str) -> bool:
    t = (text or "").lower().strip()

    if "amazon" in t and ("fba" in t or "seller central" in t):
        return True

    brand_direct_signals = [
        "brand direct", "direct brand", "wholesale", "authorized reseller",
        "commercial invoice", "brand authorization", "manufacturer",
        "distributor", "distribution", "purchase order", "po", "moq",
        "reorder", "map"
    ]
    if any(s in t for s in brand_direct_signals):
        return True

    return any(k in t for k in FBA_KEYWORDS)


# ============================
# CSV MEMORY
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
# SAFE DISCORD SENDING
# ============================
def _split_text_safely(text: str, limit: int = SAFE_SEND_LIMIT):
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return [text]

    chunks, buf = [], ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf)
        buf = ""

    for p in text.split("\n\n"):
        candidate = p if not buf else buf + "\n\n" + p
        if len(candidate) <= limit:
            buf = candidate
            continue

        if buf:
            flush()

        if len(p) <= limit:
            buf = p
            continue

        # split by lines
        line_buf = ""
        for line in p.split("\n"):
            line_candidate = line if not line_buf else line_buf + "\n" + line
            if len(line_candidate) <= limit:
                line_buf = line_candidate
            else:
                if line_buf:
                    chunks.append(line_buf)
                    line_buf = line
                else:
                    for i in range(0, len(line), limit):
                        chunks.append(line[i:i + limit])
                    line_buf = ""
        if line_buf:
            chunks.append(line_buf)

    if buf:
        flush()

    # close unbalanced code fences
    open_fence = False
    for i, c in enumerate(chunks):
        if c.count("```") % 2 == 1:
            open_fence = not open_fence
        chunks[i] = c
    if open_fence and chunks:
        chunks[-1] = chunks[-1] + "\n```"

    return chunks


async def send_long(channel: discord.abc.Messageable, text: str):
    for part in _split_text_safely(text, SAFE_SEND_LIMIT):
        if len(part) > DISCORD_CHAR_LIMIT:
            part = part[:DISCORD_CHAR_LIMIT]
        await channel.send(part)


# ============================
# OUTPUT CLEANUP (STOPS "TWO ANSWERS")
# ============================
def _strip_griffin_mentions(text: str) -> str:
    if not text:
        return text
    lines = []
    for line in text.splitlines():
        if "griffin" in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _keep_single_structured_answer(text: str) -> str:
    if not text:
        return text
    idx = text.rfind("Direct Answer")
    if idx != -1:
        return text[idx:].strip()
    return text.strip()


def finalize_reply(reply: str) -> str:
    if not reply:
        return "I can only help with Amazon FBA questions."

    reply = reply.strip()
    if reply == "I can only help with Amazon FBA questions.":
        return reply

    reply = _keep_single_structured_answer(reply)
    reply = _strip_griffin_mentions(reply)
    reply = reply.rstrip() + "\n\n" + GRIFFIN_LINE
    return reply


# ============================
# VV SYSTEM PROMPT (MORE DETAIL, STILL ONE ANSWER)
# ============================
VV_SYSTEM_PROMPT = """
You are a private Discord AI assistant for VV Sourcing.

You answer like a senior Amazon FBA Brand Direct operator who has executed wholesale at scale.
Your responses should feel like a high-quality ChatGPT explanation written by someone who understands Amazon mechanics, brand behavior, and long-term scaling.

Your goal is not just to answer — it is to *educate with context*.

========================
ANSWER STYLE (MANDATORY)
========================
Every answer must be:
- Written as ONE cohesive response
- Clear, confident, and operator-led
- Narrative and explanatory, not academic or checklist-heavy

Use this structure by default:

• Short intro paragraph framing the concept  
• 5–7 numbered sections with clear titles  
• Each section must introduce NEW insight  
• A short “Rule of Thumb” or “Bottom Line” section

Do NOT:
- Repeat the same idea in different words
- Restart the answer halfway through
- Provide multiple versions of the same answer
- Summarize the entire response again at the end

========================
DEPTH REQUIREMENTS (THIS IS KEY)
========================
Each numbered section MUST:
- Explain **why** the signal exists
- Explain **how it shows up in real Amazon data**
- Explain **what it means operationally for Brand Direct sellers**

Assume the reader already understands Amazon basics.
Your job is to explain *what experienced sellers see that beginners miss*.

Think:
“Here’s what this looks like in the real Amazon ecosystem.”

========================
BRAND DIRECT CONTEXT (ALWAYS APPLY)
========================
All explanations must be framed within Amazon FBA Brand Direct wholesale.

Brand Direct means:
- Inventory sourced directly from brands or manufacturers
- Sold on Amazon using FBA
- Supported by real invoices and authorization
- Scaled through reorders and long-term relationships

Never explain concepts using:
- Retail arbitrage logic
- Private label logic
- Dropshipping logic
- Shopify / DTC logic

If a misconception comes from arbitrage-style thinking, explicitly correct it.

========================
WHAT TO ADD (TO INCREASE INFORMATION VALUE)
========================
When relevant, layer in:
- Demand validation logic
- Seller behavior patterns
- Brand incentives and risk avoidance
- Buy Box dynamics
- MAP enforcement implications
- Reorder and stock-depth signals
- Why Amazon allows or favors certain structures

These details should feel *naturally woven in*, not bolted on.

========================
EXAMPLE MINDSET (HOW TO THINK)
========================
“If you only count sellers, you miss the point.
What matters is *who those sellers are*, *how they source*, and *how the brand supports the channel*.”

========================
GRIFFIN RULE
========================
- NEVER mention Griffin inside the answer body
- NEVER mention Discord, screenshots, or SOPs unless explicitly asked
- The system will append the Griffin line automatically

========================
OUT-OF-SCOPE HANDLING
========================
If a question is truly outside Amazon FBA or Brand Direct, reply EXACTLY:
"I can only help with Amazon FBA questions."

"""


def generate_fba_reply(history: str, message: str) -> str:
    user_input = f"Conversation history:\n{history}\n\nUser question:\n{message}"

    response = client.responses.create(
        model=MODEL_NAME,
        max_output_tokens=1400,  # ✅ more room = more detail
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
    return finalize_reply(reply)


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

    is_dm = (message.guild is None)

    if DEBUG_LOGS:
        print(
            f"[DEBUG] is_dm={is_dm} author={message.author} content={message.content!r}"
        )

    if not is_dm and REPLY_ONLY_WHEN_MENTIONED_IN_SERVER:
        if bot.user not in message.mentions:
            return

    text = (message.content or "").strip()
    if not is_dm:
        text = strip_mention(text, bot.user.id)

    if not text:
        await send_long(message.channel,
                        "Mention me with your Amazon FBA question.")
        return

    user_id = str(message.author.id)
    history = load_user_history(user_id, HISTORY_LIMIT)

    if not is_amazon_fba_question(text):
        combined = (history + "\nUser: " + text).lower()
        if not is_amazon_fba_question(combined):
            await send_long(message.channel,
                            "I can only help with Amazon FBA questions.")
            return

    save_turn(user_id, "user", text)

    try:
        reply = generate_fba_reply(history, text)
    except Exception as e:
        err = str(e)
        print("[ERROR] OpenAI call failed:", err)

        if "insufficient_quota" in err or "Error code: 429" in err:
            await send_long(
                message.channel,
                "⚠️ OpenAI quota/billing issue (insufficient quota). Check your OpenAI billing."
            )
        elif "model" in err and ("not found" in err
                                 or "does not exist" in err):
            await send_long(
                message.channel,
                "⚠️ Model not available. Use MODEL_NAME='gpt-4o-mini'.")
        else:
            await send_long(message.channel,
                            f"⚠️ Error generating reply: {err}")
        return

    save_turn(user_id, "assistant", reply)
    await send_long(message.channel, reply)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
