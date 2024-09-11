import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from typing import Dict, Any
import asyncio

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
            for event in self.graph.stream({"messages": [("human", message)]}, config):
                logger.debug(f"Received event: {event}")
                if "messages" in event:
                    for msg in event["messages"]:
                        logger.debug(f"Processing message: {msg}")
                        response_content = self.extract_content(msg)
                        if response_content:
                            await update.message.reply_text(response_content)
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await update.message.reply_text("I'm sorry, but I encountered an error while processing your request.")

    def extract_content(self, message):
        logger.debug(f"Extracting content from: {message}")
        if isinstance(message, tuple):
            if len(message) == 2 and message[0] == "ai":
                return str(message[1])
            else:
                return str(message)
        elif isinstance(message, dict):
            return str(message.get('content', message))
        elif isinstance(message, str):
            return message
        else:
            return str(message)

    def run(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
