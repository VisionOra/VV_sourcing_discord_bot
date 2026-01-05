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

# Discord message length limit
DISCORD_CHAR_LIMIT = 2000
SAFE_SEND_LIMIT = 1900  # leave buffer for code fences / extra newlines

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

    # strong Amazon signals
    if "amazon" in t and ("fba" in t or "seller central" in t):
        return True

    # treat Brand Direct / wholesale language as in-scope even if "amazon" not typed
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
# SAFE DISCORD SENDING (SPLITS > 2000)
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

    paragraphs = text.split("\n\n")
    for p in paragraphs:
        candidate = p if not buf else buf + "\n\n" + p

        if len(candidate) <= limit:
            buf = candidate
            continue

        if buf:
            flush()

        if len(p) <= limit:
            buf = p
            continue

        # Paragraph too big: split by lines
        lines = p.split("\n")
        line_buf = ""
        for line in lines:
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

    # Best-effort code fence fix
    open_fence = False
    for i, c in enumerate(chunks):
        if c.count("```") % 2 == 1:
            open_fence = not open_fence
        chunks[i] = c
    if open_fence and chunks:
        chunks[-1] = chunks[-1] + "\n```"

    return chunks


async def send_long(channel: discord.abc.Messageable, text: str):
    parts = _split_text_safely(text, SAFE_SEND_LIMIT)
    for part in parts:
        if len(part) > DISCORD_CHAR_LIMIT:
            part = part[:DISCORD_CHAR_LIMIT]
        await channel.send(part)


# ============================
# ENFORCE GRIFFIN LINE ONLY ONCE (AT END)
# ============================
def ensure_single_griffin_line(reply: str) -> str:
    if not reply:
        return reply

    if reply.strip() == "I can only help with Amazon FBA questions.":
        return reply.strip()

    # Remove any existing Griffin line occurrences
    cleaned_lines = []
    for line in reply.splitlines():
        if line.strip() == GRIFFIN_LINE.strip():
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    # Append once at end
    return cleaned.rstrip() + "\n\n" + GRIFFIN_LINE


# ============================
# VV SOURCING SYSTEM PROMPT
# (IMPORTANT: no instruction telling the model to write the Griffin line)
# ============================
VV_SYSTEM_PROMPT = """
You are a private Discord AI assistant for VV Sourcing.

You answer questions the way an experienced Amazon FBA Brand Direct consultant would — clear, strategic, and educational. Your responses should resemble ChatGPT-style explanations: layered, contextual, and insightful, not short or surface-level.

You are NOT a casual chatbot. You are an operator-level assistant focused on long-term, compliant Amazon wholesale success.

=================================================
ANSWER STYLE (MAKE IT "MORE INFO" LIKE CHATGPT)
=================================================
For every in-scope question, follow this structure:

1) Direct Answer (1–2 sentences)
2) Why It Works (explain reasoning and business logic)
3) What Brands/Amazon Look For (real-world criteria and signals)
4) Common Mistakes / Misconceptions (brief)
5) Practical Next Steps (2–5 actionable steps)
6) If helpful: a short example, checklist, or mini-framework

Depth rules:
- Cover at least 3 meaningful angles per answer (ops + compliance + scaling).
- Use concrete examples (MAP, invoices, reorders, seller behavior, in-stock rate, etc.) when relevant.
- If the question is broad, ask at most 2 clarifying questions AFTER giving a strong default answer.
- Do not ramble. Be dense with insight.

=================================================
BRAND DIRECT (CORE MODEL)
=================================================
Amazon Brand Direct is a wholesale distribution model, not a tactic.

Definition:
- Source inventory directly from brands or manufacturers
- Sell that inventory on Amazon using Amazon FBA
- NOT retail/online arbitrage, NOT dropshipping, NOT invoice fabrication

Required standards:
- Legit wholesale pricing
- Real commercial invoices
- Brand authorization to sell
- Clean supply chain documentation
- Inventory flows brand → Amazon FBA → customer
- Scale through repeat POs and reorders

=================================================
PRODUCT RESEARCH SOPs
=================================================
A product is viable ONLY if it meets these rules:
- More than 3 FBA sellers
- At least 2 new FBA sellers in past 6 months
- 100+ units/month (seasonal can use last year)
- 2–3 listings total invoice value > $3k–$4k, or 1 listing > $4k
- No IP complaint history
- Weight limits: under $100 max 5 lbs; over $100 max 7 lbs
- New seller should have 30+ units in stock
- Do not proceed if the brand is selling on Amazon

=================================================
IN/OUT OF SCOPE
=================================================
You MAY answer:
- Amazon FBA operations, Brand Direct, wholesale sourcing, compliance, brand approval logic

You MAY NOT answer:
- Retail arbitrage, online arbitrage, dropshipping, Shopify/DTC tactics, paid ads/funnels, invoice fabrication

=================================================
OUT-OF-SCOPE HANDLING
=================================================
If truly outside Amazon FBA or Brand Direct, reply EXACTLY:
"I can only help with Amazon FBA questions."
"""


def generate_fba_reply(history: str, message: str) -> str:
    user_input = f"Conversation history:\n{history}\n\nUser question:\n{message}"

    response = client.responses.create(
        model=MODEL_NAME,
        max_output_tokens=900,
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

    if not reply:
        return "I can only help with Amazon FBA questions."

    # Ensure Griffin line appears exactly once at end (and nowhere else)
    return ensure_single_griffin_line(reply)


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

    # Server: reply only if mentioned
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

    # FBA-only gate (history-aware follow-ups)
    if not is_amazon_fba_question(text):
        combined = (history + "\nUser: " + text).lower()
        if not is_amazon_fba_question(combined):
            await send_long(message.channel,
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

    # Send safely (prevents 2000 char error)
    await send_long(message.channel, reply)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
