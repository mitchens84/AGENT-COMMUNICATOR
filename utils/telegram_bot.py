import logging
import traceback
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from typing import Dict, Any
import asyncio
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, token: str, graph):
        self.application = Application.builder().token(token).build()
        self.graph = graph

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text("Hello! I'm a research assistant. How can I help you today?")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return

        message = update.message.text
        user_id = update.effective_user.id

        config = {"configurable": {"thread_id": str(user_id)}}

        try:
            logger.debug(f"Received message: {message}")
            refined_query = ""
            search_results = ""
            for event in self.graph.stream({"messages": [HumanMessage(content=message)]}, config):
                logger.debug(f"Received event: {event}")
                for key, value in event.items():
                    if "messages" in value:
                        for msg in value["messages"]:
                            if isinstance(msg, AIMessage):
                                if msg.content.startswith("Refined query:"):
                                    refined_query = msg.content
                                elif msg.content.startswith("Search results:"):
                                    search_results = msg.content

            logger.debug(f"Refined query: {refined_query}")
            logger.debug(f"Search results: {search_results}")

            response_text = f"{refined_query}\n\n{search_results}".strip()
            if response_text:
                # Split the response if it's too long
                max_length = 4096  # Telegram's max message length
                response_parts = [response_text[i:i+max_length] for i in range(0, len(response_text), max_length)]
                for part in response_parts:
                    await update.message.reply_text(part.strip())
            else:
                logger.warning("No response generated")
                await update.message.reply_text("I'm sorry, but I couldn't generate a response. Please try again.")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(traceback.format_exc())
            await update.message.reply_text("I'm sorry, but I encountered an error while processing your request. Please try again later.")

    def run(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.run_polling()
