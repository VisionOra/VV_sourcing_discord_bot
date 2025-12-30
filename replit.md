# Discord FAQ Bot

## Overview
A Discord bot that uses LangChain and OpenAI to answer questions based on a CSV knowledge base (FAQ).

## Project Structure
- `app.py` - Main bot application
- `New FAQ.csv` - Knowledge base data
- `requirements.txt` - Python dependencies

## Tech Stack
- Python 3.11
- Discord.py - Discord bot framework
- LangChain - AI/LLM framework
- OpenAI - Embeddings and LLM
- FAISS - Vector store for similarity search

## Required Environment Variables
- `OPENAI_API_KEY` - OpenAI API key for embeddings and completions
- `DISCORD_BOT_TOKEN` - Discord bot token

## Running the Bot
The bot runs via the "Discord Bot" workflow with `python app.py`.

## Bot Commands
- `!ask <question>` - Ask a question that will be answered using the FAQ knowledge base
