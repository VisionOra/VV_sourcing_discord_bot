import os
import discord
from discord.ext import commands
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read secrets from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Setup Discord intents properly
intents = discord.Intents.default()
intents.message_content = True  # ✅ This must be True for message reading
bot = commands.Bot(command_prefix="!", intents=intents)

# Load the knowledge base
def load_knowledge_base():
    try:
        loader = CSVLoader(file_path="New FAQ.csv", encoding="utf-8")
        documents = loader.load()
    except Exception:
        loader = CSVLoader(file_path="New FAQ.csv", encoding="ISO-8859-1")
        documents = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Initialize vectorstore and QA chain
vectorstore = load_knowledge_base()
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY),
    retriever=vectorstore.as_retriever()
)

# Bot ready event
@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

# Ask command
@bot.command(name='ask')
async def ask(ctx, *, question: str):
    print(f"❓ Received question: {question}")
    await ctx.send("🤖 Processing your question...")
    try:
        response = qa_chain.run(question)
        await ctx.send(response)
    except Exception as e:
        await ctx.send(f"⚠️ Error: {str(e)}")

# Run the bot
if __name__ == "__main__":
    if not DISCORD_TOKEN or not OPENAI_API_KEY:
        raise ValueError("Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY in environment.")
    bot.run(DISCORD_TOKEN)
