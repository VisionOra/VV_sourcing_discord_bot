import os
import discord
from discord.ext import commands
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

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

vectorstore = load_knowledge_base()
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(openai_api_key=OPENAI_API_KEY),
    retriever=vectorstore.as_retriever()
)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    print(f"Received question: {question}")
    await ctx.send("Processing your question...")
    try:
        response = qa_chain.run(question)
        await ctx.send(response)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")

if __name__ == "__main__":
    if not DISCORD_TOKEN or not OPENAI_API_KEY:
        raise ValueError("Missing DISCORD_BOT_TOKEN or OPENAI_API_KEY in environment.")
    bot.run(DISCORD_TOKEN)
