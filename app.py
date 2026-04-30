import os
import csv
import json
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
EMAIL_TEMPLATES_FILE = "email_templates.json"
MEMORY_FILE = "user_memory.csv"
MODEL_NAME = "gpt-4o-mini"
REPLY_ONLY_WHEN_MENTIONED_IN_SERVER = True
DEBUG_LOGS = True

DISCORD_CHAR_LIMIT = 2000
SAFE_SEND_LIMIT = 1900

# ============================
# EMAIL HELP MENU
# ============================
EMAIL_HELP_TEXT = """**GriffinBot — Email Assistant**

I can write professional outreach emails for you. Just describe what you need naturally.

**How to use:**
> `@GriffinBot send a follow up email to Nike, no response in 2 weeks`
> `@GriffinBot write a follow up to Adidas about our wholesale inquiry`
> `@GriffinBot follow up with New Balance, they haven't replied`

**Available templates:**
> `follow_up_1` — Short polite nudge
> `follow_up_2` — Follow up referencing wholesale terms and product availability
> `follow_up_3` — Follow up with partnership pitch, request a call
> `follow_up_4` — Casual follow up, asks for alternate contact
> `follow_up_5` — Follow up on wholesale partnership, asks for next steps

**Tip:** You don't need to type the template name — just describe the situation and I'll pick the right one.
"""

EMAIL_INTENT_KEYWORDS = [
    "email",
    "outreach",
    "reach out",
    "contact",
    "write to",
    "send a message",
    "follow up",
    "follow-up",
    "followup",
    "authorization",
    "introduce",
    "partnership email",
    "send an email",
    "draft",
    "compose",
    # initial outreach triggers
    "first email",
    "initial email",
    "cold email",
    "short email",
    "business order",
    "order inquiry",
    "business inquiry",
    "wholesale inquiry",
    "inquiry to",
    "interested in",
    "asking for pricing",
    "asking about pricing",
    "pricing inquiry",
    "distributor inquiry",
    "want to partner",
    "looking to partner",
    "brand inquiry",
    "product inquiry",
    # reply/response scenarios
    "reply back",
    "reply to",
    "respond to",
    "write back",
    "get back to",
    "said they",
    "they said",
    "they replied",
    "they responded",
    "can you reply",
    "can you write",
    "can you respond",
    "rejection",
    "rejected",
    "turned us down",
    "not interested",
    "no response",
    "haven't replied",
    "didn't reply",
    # brand response descriptions
    "not taking on",
    "not working with",
    "not accepting",
    "said no",
    "said we",
    "said they don't",
    "said they are not",
    "said they're not",
    "told us",
    "told me",
]

BRAND_RESPONSE_PATTERNS = [
    'said "',
    "said '",
    "they said",
    "brand said",
    "she said",
    "he said",
    "told me",
    "told us",
    "their response",
    "they responded",
    "they replied",
    "not taking on",
    "not accepting",
    "not working with",
    "we are not",
    "we don't",
    "we do not",
]


def is_email_intent(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in EMAIL_INTENT_KEYWORDS)


def _is_brand_response(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in BRAND_RESPONSE_PATTERNS)


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
                        chunks.append(line[i : i + limit])
                    line_buf = ""
        if line_buf:
            chunks.append(line_buf)

    if buf:
        flush()

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
# EMAIL AGENT
# ============================

_email_sessions: dict = {}

PASTED_TEMPLATE_SIGNALS = [
    "subject:",
    "hi [",
    "dear [",
    "hello [",
    "[brand]",
    "[name]",
    "[repname]",
    "[productname]",
    "[companyname]",
    "[existingbrands]",
]

EMAIL_AGENT_SYSTEM = """You are Griffin, a friendly and slightly informal email assistant for a brand-direct wholesale distributor.

Your job: help users write professional outreach emails to brands, answer outreach questions, and refine emails on request.

========================
PERSONALITY
========================
- Warm, direct, slightly casual — like a helpful teammate, not a corporate form
- Never stiff or overly formal in your own messages (the emails you write can be formal)
- Short sentences. Get to the point. No fluff.

========================
[CompanyName] RESOLUTION RULES
========================
Templates use [CompanyName] as a placeholder. Resolve it using these rules before writing any email:

1. SINGLE COMPANY — if the user mentions one company name (their own), use it exactly.
   Example: user says "write from VV Sourcing" → [CompanyName] = "VV Sourcing"

2. MULTIPLE COMPANIES — if the user mentions multiple sender companies, treat them as Company 1, Company 2, etc. and write separate emails or iterate clearly.
   Example: user says "write for VV Sourcing and ABC Distributors" → write email for Company 1 (VV Sourcing), then Company 2 (ABC Distributors)

3. PARTNERSHIP EMAIL — if the email is a co-outreach or partnership pitch involving two company names, format as: [YourCompany] x [PartnerCompany]
   Example: "VV Sourcing x Nike" in the subject line and opening
   Use [company name] for the sender side if no name is given: "[company name] x [Brand]"

4. UNKNOWN — if the user provides no company name at all, leave it as [company name] — never invent a name, never default to "VV Sourcing".

========================
CONVERSATION FLOWS
========================

1. ANSWERING QUESTIONS
If the user asks a general question about outreach, subject lines, email best practices, or our process — answer it directly and concisely. Offer to help write something if relevant.

Example:
User: what's a good subject line for cold outreach?
Griffin: Keep it 4–7 words. Specific beats clever — "Partnership Inquiry - [Brand] x [CompanyName]" works better than "Exciting Opportunity". Want me to generate a few options?

2. GENERATING AN EMAIL (vague request)
If the user says "write an email" or "generate an outreach" without enough detail, ask ONE clarifying question to move forward. Do not ask for everything at once.

Example:
User: generate an outreach email
Griffin: Sure — is this an initial outreach, a follow-up, or a breakup email?
User: Initial
Griffin: Got it. Which brand, and anything specific you want to lead with?

3. GENERATING AN EMAIL (enough context provided)
If the user gives enough info (brand name + situation), generate the email immediately without asking ANYTHING — no subject line, no contact name, no clarifying questions. Fill what you can from context, use [MISSING: value] for anything unknown except company name — always use [company name] for that, and output the email right away.

4. PASTED TEMPLATE
If the user pastes a template with placeholders like [Brand], [Name], etc., fill in everything you can from context. Mark missing values as [MISSING: value], except company name — always use [company name] for that. Never skip this step.

5. REFINEMENT
If the user asks to tweak a previous email ("make it shorter", "more casual", "add a P.S."), apply the change and return the full revised email. Do not re-ask what they want changed.

========================
BRAND REJECTION HANDLING
========================
If the user shares that a brand said no or rejected them (e.g. "they said they're not taking on Amazon sellers", "brand said not interested"), this IS in scope.

Your goal: write a reply that makes them reconsider. Not defend [CompanyName] — make the brand feel like they might be making a mistake by closing the door too quickly.

STRUCTURE — follow this exactly:

IMPORTANT: Keep the greeting (Hi [Name], / Hi [Brand] Team,) and sign-off (Best regards, / Best,) exactly as they appear in the matched template. Only rewrite the body between them.

Body Paragraph 1 — Acknowledge + disarm:
- One short sentence that accepts their decision without pushback
- Immediately pivot to "before we close the door" — plant a seed of doubt
- Example: "Appreciate you letting me know. I'd love the chance to share a bit more about how I operate before we close the door — I think my approach may be different from what you've experienced with other [objection type] sellers."

Body Paragraph 2 — Reframe who you are (address their specific concern):
- 2–3 short sentences that speak directly to their worry
- If "not taking Amazon sellers" → talk about being selective with brands, protecting MAP pricing, not competing on price in ways that erode brand value, being an asset not a liability
- If "not interested" → talk about the incremental revenue channel with zero upfront risk, no obligation
- If "not working with distributors" → talk about maintaining brand control, transparency, treating the brand as a partner not just a vendor
- Make it feel personal and specific, not like a pitch deck

Body Paragraph 3 — Low-friction close:
- Ask for a short call (10–15 min), not a commitment
- Frame it as "let me show you, not tell you" — if it's still not a fit after, that's fine
- Example: "Would you be open to a quick 15-minute call? I'd rather show you what I do than just tell you — and if it's still not a fit after that, I completely understand."

TONE RULES — these are non-negotiable:
- Write exactly like this example body (confident, human, no corporate fluff):
  "Appreciate you letting me know. I'd love the chance to share a bit more about how I operate before we close the door — I think my approach may be different from what you've experienced with other Amazon sellers.

  I'm selective about the brands I work with, I don't compete on price in ways that erode brand value, and I actively protect MAP pricing. My goal is always to be an asset to a brand's presence on Amazon, not a liability.

  Would you be open to a quick 15-minute call? I'd rather show you what I do than just tell you — and if it's still not a fit after that, I completely understand."
- Short sentences. Conversational. No buzzwords.
- Never use: "I appreciate your response", "I hope this finds you well", "Looking forward to your prompt response", "at your earliest convenience", "I wanted to reach out"
- The email must feel written for this specific brand and their specific objection — not templated

Always show the template used.

========================
EMAIL WRITING STYLE (ALL EMAILS — initial outreach, follow-up, rebuttal)
========================
NEVER copy the template body word for word. The template is a skeleton — you write the email fresh, using the template only for structure and key details (numbers, intent).

VOICE: Write like a senior salesperson who has closed hundreds of deals. Confident, direct, no filler. Every sentence either builds credibility, creates desire, or drives action. Nothing else belongs in the email.

MINIMUM LENGTH: The body (excluding greeting and sign-off) must be at least 4 solid paragraphs. A 2-sentence email is a failure. If you write fewer than 4 paragraphs, rewrite it.

Every email body must have these 4 paragraphs:

Paragraph 1 — Credibility + specific reason for reaching out:
- Don't open with "We're big fans" or "I hope you're doing well" — open with a statement of who you are and why it's relevant to THEM
- Lead with what [CompanyName] does and the scale we operate at
- Make it clear this isn't a cold shot in the dark — there's a specific reason you're reaching out to this brand
- Example (initial outreach): "[CompanyName] is a full-line distributor with an active buyer base across Amazon and other major retail channels. We move volume — and we're selective about the brands we bring on. [Brand]'s [product] is exactly the kind of product our customers are already looking for."

Paragraph 2 — Concrete value, no vague promises:
- Give them numbers: initial order potential, monthly volume, revenue range (use numbers from the template if available — $200K–$300K, 500+ units)
- Tell them what they get: incremental revenue, brand-safe distribution, MAP protection, reach into buyer segments they may not be accessing
- Write this like you're laying chips on the table — matter-of-fact, not pleading
- Example: "We're prepared to place an initial order of 500+ units, with monthly volume that can scale to $200K–$300K. We enforce MAP pricing and manage listings to protect brand integrity — so you're not trading margin for reach."

Paragraph 3 — Why us / why now (social proof + urgency):
- 2–3 sentences. Name-drop comparable brands we work with or categories we dominate (use [ExistingBrands] from template if available, otherwise reference the category)
- Create mild urgency — we're expanding in this category right now, we're onboarding new brands this quarter, our buyers are actively asking for this type of product
- Example: "We currently work with brands like [ExistingBrands] across similar categories and our buyers are actively searching for more options in this space. We're expanding our tools portfolio this quarter and [Brand] is at the top of our list."

Paragraph 4 — Direct ask, no hedging:
- One sentence. Tell them exactly what you need: pricing sheet, wholesale terms, a 15-min call.
- Don't soften it with "if you have a moment" or "whenever you get a chance"
- Example: "Could you send over your wholesale pricing and distributor terms so we can move this forward?"

TONE RULES — non-negotiable:
- Write like someone who doesn't need the deal, but sees a clear opportunity
- Short sentences. No passive voice. No corporate filler.
- BANNED: "We're big fans", "I hope this finds you well", "looking forward to exploring", "I wanted to reach out", "please don't hesitate", "at your earliest convenience", "we would love the opportunity", "we're excited about the possibility"
- One strong specific sentence beats three vague enthusiastic ones
- Always replace [Brand], [ProductName], [ExistingBrands] with the actual values from context — never leave bare brackets
- If a value is genuinely unknown, leave it as a plain bracket placeholder: [rep name], [company name], [product name] — never use [MISSING: ...] format

========================
OUT OF SCOPE
========================
If the user asks for something that does NOT match any template in the library (e.g. complaint emails, invoice disputes, customer service issues), do NOT attempt to write it.
Instead, tell them clearly what you can write, listing the available templates by name from the TEMPLATE LIBRARY section above.
Say something like: "I can only write outreach and follow-up emails to brands. Here's what I have: [list template names]. Want me to write one of these?"

========================
OUTPUT RULES
========================
- When you generate an email from the template library, ALWAYS start your reply with a single line: `📋 Template used: [template_id] — Template Name` then a blank line, then the email
- If the user pasted their own template, start with: `📋 Template used: custom (user-pasted)`
- Always wrap finished emails in a ```email code block
- After every email, offer 1-2 short refinement options on a new line (e.g. "Want it shorter or more formal?")
- Never ask more than one clarifying question at a time
- If you already have the context from earlier in the conversation, don't re-ask for it
- Keep your own conversational messages short — save the length for the email

========================
TEMPLATE LIBRARY (use these as structural guides — always write with a human, persuasive voice, never copy verbatim)
========================
{template_context}
"""


def load_all_templates() -> list:
    if not os.path.exists(EMAIL_TEMPLATES_FILE):
        return []
    with open(EMAIL_TEMPLATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


_TEMPLATES: list = []
_TOOLS: list = []  # function calling tools — descriptions only, no bodies
_TEMPLATE_MAP: dict = {}  # id → full template object for fast lookup


def _ensure_templates_loaded():
    global _TEMPLATES, _TOOLS, _TEMPLATE_MAP
    if not _TEMPLATES:
        _TEMPLATES = load_all_templates()
        _TEMPLATE_MAP = {t["id"]: t for t in _TEMPLATES}
        _TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": t["id"],
                    "description": t["description"],
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
            for t in _TEMPLATES
        ]


def _init_csv():
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts", "user_id", "role", "content"])


def _get_email_history(user_id: str) -> list:
    # Check in-memory cache first
    if user_id in _email_sessions:
        return _email_sessions[user_id]
    # Load from CSV on first access (e.g. after bot restart)
    if not os.path.exists(MEMORY_FILE):
        return []
    rows = []
    with open(MEMORY_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("user_id") == user_id:
                rows.append({"role": row["role"], "content": row["content"]})
    rows = rows[-20:]
    _email_sessions[user_id] = rows
    return rows


def _save_email_turn(user_id: str, role: str, content: str):
    _init_csv()
    # Update in-memory cache
    if user_id not in _email_sessions:
        _email_sessions[user_id] = []
    _email_sessions[user_id].append({"role": role, "content": content})
    _email_sessions[user_id] = _email_sessions[user_id][-20:]
    # Persist to CSV
    with open(MEMORY_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([int(time.time()), user_id, role, content])


def clear_email_session(user_id: str):
    _email_sessions.pop(user_id, None)
    # Remove user rows from CSV
    if not os.path.exists(MEMORY_FILE):
        return
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("user_id") != user_id]
    with open(MEMORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ts", "user_id", "role", "content"])
        writer.writeheader()
        writer.writerows(rows)


def _is_pasted_template(text: str) -> bool:
    t = text.lower()
    return sum(1 for s in PASTED_TEMPLATE_SIGNALS if s in t) >= 2


def _is_refinement(user_message: str, history: list) -> bool:
    return bool(history) and any(
        kw in user_message.lower()
        for kw in [
            "shorter",
            "longer",
            "casual",
            "formal",
            "p.s.",
            "add",
            "remove",
            "change",
            "tweak",
            "rewrite",
            "regenerate",
            "different",
            "more",
            "less",
            "make it",
            "can you",
            "try",
            "again",
        ]
    )


def _is_question(user_message: str) -> bool:
    t = user_message.lower().strip()
    return t.endswith("?") or any(
        t.startswith(w)
        for w in [
            "what",
            "how",
            "why",
            "when",
            "which",
            "who",
            "is ",
            "are ",
            "can ",
            "do ",
            "does ",
        ]
    )


# ── CALL 1: function calling router ─────────────────────────────
def _route_template(user_message: str, history: list) -> dict | None:
    """Send only descriptions as tools. GPT returns the best matching template ID."""
    # Build a compact conversation summary for routing context
    history_summary = ""
    if history:
        history_summary = "\n\nRecent conversation:\n" + "\n".join(
            f"{'User' if m['role'] == 'user' else 'Griffin'}: {m['content'][:120]}"
            for m in history[-4:]
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an email routing assistant for a brand-direct wholesale distributor. Based on the user's request, call the function that best matches the email they need. If nothing matches, do not call any function.",
            },
            {"role": "user", "content": user_message + history_summary},
        ],
        tools=_TOOLS,
        tool_choice="auto",  # allows GPT to call nothing if no match
        max_tokens=50,
    )

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return None

    chosen_id = tool_calls[0].function.name  # type: ignore[union-attr]
    if DEBUG_LOGS:
        print(f"[ROUTER] Chose → {chosen_id}")

    return _TEMPLATE_MAP.get(chosen_id)


# ── CALL 2: Griffin filler / conversational handler ─────────────
def _griffin_respond(
    user_message: str, history: list, template: dict | None = None, extra_hint: str = ""
) -> str:
    """Griffin handles filling, Q&A, refinement, and out-of-scope — with only the matched template in context."""

    if template:
        template_section = (
            f"MATCHED TEMPLATE: [{template['id']}] {template['name']}\n"
            f"Body:\n{template.get('template', '')}"
        )
    else:
        # No template matched — give Griffin just the names list, not the bodies
        template_section = "Available templates (names only):\n" + "\n".join(
            f"- `{t['id']}` — {t['name']}" for t in _TEMPLATES
        )

    system = EMAIL_AGENT_SYSTEM.replace("{template_context}", template_section)

    content = extra_hint + user_message if extra_hint else user_message

    messages = (
        [{"role": "system", "content": system}]
        + history
        + [{"role": "user", "content": content}]
    )

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=1000
    )
    return (response.choices[0].message.content or "").strip()


# ── Main entry point ─────────────────────────────────────────────
def generate_email_reply(user_id: str, user_message: str) -> str:
    _ensure_templates_loaded()
    history = _get_email_history(user_id)

    if DEBUG_LOGS:
        print(f"[GRIFFIN] history_turns={len(history)} message={user_message[:60]!r}")

    _save_email_turn(user_id, "user", user_message)

    # 1. User pasted their own template — skip routing entirely
    if _is_pasted_template(user_message):
        if DEBUG_LOGS:
            print("[GRIFFIN] Detected pasted template")
        hint = "[NOTE: The user pasted their own template. Fill all placeholders from context, mark missing as [MISSING: value] except company name which must be left as [company name], ask if they need changes.]\n\n"
        reply = _griffin_respond(user_message, history, template=None, extra_hint=hint)
        _save_email_turn(user_id, "assistant", reply)
        return reply

    # 2. Refinement of previous email — skip routing, Griffin has history
    if _is_refinement(user_message, history):
        if DEBUG_LOGS:
            print("[GRIFFIN] Detected refinement")
        reply = _griffin_respond(user_message, history, template=None)
        _save_email_turn(user_id, "assistant", reply)
        return reply

    # 3. General outreach question — skip routing
    if _is_question(user_message) and not is_email_intent(user_message):
        if DEBUG_LOGS:
            print("[GRIFFIN] Detected question")
        reply = _griffin_respond(user_message, history, template=None)
        _save_email_turn(user_id, "assistant", reply)
        return reply

    # 4. CALL 1 — function calling router picks the template
    template = _route_template(user_message, history)

    # 5. CALL 2 — Griffin fills the template or handles no-match
    hint = ""
    if template:
        hint = "[NOTE: A template has been matched. Generate the full email immediately — do NOT ask any clarifying questions. Fill placeholders from context, use [MISSING: value] for unknown fields except company name — always use [company name] for that.]\n\n"
    reply = _griffin_respond(user_message, history, template=template, extra_hint=hint)
    _save_email_turn(user_id, "assistant", reply)
    return reply


# ============================
# DISCORD BOT
# ============================
intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    _init_csv()
    print(f"✅ Logged in as {bot.user}")


@bot.command(name="resetemail")
async def resetemail(ctx):
    clear_email_session(str(ctx.author.id))
    await ctx.send("✅ Email conversation reset. Start fresh anytime!")


def strip_mention(text: str, bot_id: int) -> str:
    return text.replace(f"<@{bot_id}>", "").replace(f"<@!{bot_id}>", "").strip()


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    is_dm = message.guild is None

    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.invoke(ctx)
        return

    if DEBUG_LOGS:
        print(
            f"[DEBUG] is_dm={is_dm} author={message.author} content={message.content!r}"
        )  # type: ignore[str-format]

    if not is_dm and REPLY_ONLY_WHEN_MENTIONED_IN_SERVER:
        if bot.user not in message.mentions:
            return

    text = (message.content or "").strip()
    if not is_dm and bot.user is not None:
        text = strip_mention(text, bot.user.id)

    if not text:
        await send_long(
            message.channel,
            "Hey! Describe the email you need and I'll write it. Type `email_help` to see options.",
        )
        return

    if text.lower().strip() in ("email_help", "email help", "emailhelp"):
        await send_long(message.channel, EMAIL_HELP_TEXT)
        return

    user_id = str(message.author.id)
    email_active = bool(_get_email_history(user_id))

    if is_email_intent(text) or _is_brand_response(text) or email_active:
        try:
            reply = generate_email_reply(user_id, text)
        except Exception as e:
            err = str(e)
            print("[ERROR] Email agent failed:", err)
            reply = f"⚠️ Error generating email: {err}"
        await send_long(message.channel, reply)
        return

    await send_long(
        message.channel,
        "I can help you write outreach emails! Describe what you need or type `email_help` to see options.",
    )


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
