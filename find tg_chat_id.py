from telethon import TelegramClient, events, sync
import os
from dotenv import load_dotenv

load_dotenv()

api_id = int(os.getenv("TELEGRAM_API_ID"))
api_hash = os.getenv("TELEGRAM_API_HASH")
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

client = TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token)

@client.on(events.NewMessage)
async def handler(event):
    print("Chat ID:", event.chat.id)
    print("Chat title:", event.chat.title)
    await client.disconnect()

client.run_until_disconnected()
