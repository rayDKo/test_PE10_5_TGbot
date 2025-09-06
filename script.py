import os
import asyncio  # для запуска асинхронных функций в синхронном коде
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from telethon import TelegramClient
from io import BytesIO
import base64  # для декодирования base64 из Stability AI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Получение ключей из переменных окружения (установите в PyCharm Run/Debug Configurations)
openai.api_key = os.getenv("OPENAI_API_KEY")
currentsapi_key = os.getenv("CURRENTS_API_KEY")
stability_api_key = os.getenv("STABILITY_API")
telegram_api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
telegram_api_hash = os.getenv("TELEGRAM_API_HASH")
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

if not all([openai.api_key, currentsapi_key, stability_api_key,
            telegram_api_id, telegram_api_hash, telegram_bot_token, telegram_chat_id]):
    raise ValueError("Установите переменные окружения: OPENAI_API_KEY, CURRENTS_API_KEY, STABILITY_API, "
                     "TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")


class Topic(BaseModel):
    topic: str


def get_recent_news(topic: str) -> str:
    url = "https://api.currentsapi.services/v1/latest-news"
    params = {"language": "en", "keywords": topic, "apiKey": currentsapi_key}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {response.text}")

    news_data = response.json().get("news", [])
    if not news_data:
        return "Свежих новостей не найдено."

    return "\n".join([article["title"] for article in news_data[:5]])


def generate_content(topic: str) -> dict:
    recent_news = get_recent_news(topic)
    try:
        title = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Придумайте привлекательный заголовок для статьи на тему '{topic}', с учётом новостей:\n{recent_news}. "
                           f"Исключить спецсимволы типа #, !, _, (, ), -"
            }],
            max_tokens=60, temperature=0.5, stop=["\n"]
        ).choices[0].message.content.strip()

        meta_description = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Напишите информативное мета-описание для статьи с заголовком '{title}'. Исключить спецсимволы."
            }],
            max_tokens=120, temperature=0.5, stop=["."]
        ).choices[0].message.content.strip()

        post_content = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Напишите подробную статью на тему '{topic}', используя новости:\n{recent_news}.
                Структура: введение, часть, заключение, не менее 1500 символов, без спецсимволов."""
            }],
            max_tokens=1000, temperature=0.5, presence_penalty=0.6, frequency_penalty=0.6
        ).choices[0].message.content.strip()

        return {"title": title, "meta_description": meta_description, "post_content": post_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации контента: {str(e)}")


def generate_image_with_stability(prompt: str, width=512, height=512) -> Image.Image:
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {"Content-Type": "application/json", "Accept": "application/json", "Authorization": f"Bearer {stability_api_key}"}
    json_data = {
        "width": width, "height": height, "samples": 1, "steps": 25, "cfg_scale": 7, "style_preset": "photographic",
        "text_prompts": [{"text": prompt, "weight": 1}]
    }
    response = requests.post(url, headers=headers, json=json_data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации изображения: {response.text}")
    data = response.json()
    base64_img = data["artifacts"][0]["base64"]
    image_bytes = BytesIO(base64.b64decode(base64_img))
    return Image.open(image_bytes).convert("RGBA")


def create_image_with_blur_and_text(img: Image.Image, title: str, output_path='story_text.jpg'):
    width, height = img.size
    blur_height = height // 4
    bottom_part = img.crop((0, height - blur_height, width, height))

    blurred = bottom_part.filter(ImageFilter.GaussianBlur(radius=10))

    gradient = Image.new('L', (width, blur_height))
    for y in range(blur_height):
        alpha = int(255 * (y / blur_height))
        for x in range(width):
            gradient.putpixel((x, y), alpha)
    blurred.putalpha(gradient)

    img.paste(blurred, (0, height - blur_height), blurred)

    draw = ImageDraw.Draw(img)
    font_path = 'Roboto-VariableFont_wdth,wght.ttf'
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)

    lines, line = [], ""
    margin = 20
    max_width = width - 2 * margin
    for word in title.split():
        test_line = f"{line} {word}".strip()
        if draw.textlength(test_line, font=font) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    total_text_height = line_height * len(lines)
    y_text = height - blur_height + (blur_height - total_text_height) // 2

    for line in lines:
        text_width = draw.textlength(line, font=font)
        x_text = (width - text_width) // 2
        draw.text((x_text, y_text), line, font=font, fill=(255, 0, 0))
        y_text += line_height

    img.convert("RGB").save(output_path, format="JPEG")


# Инициализация клиента Telethon
client = TelegramClient('bot_session', telegram_api_id, telegram_api_hash).start(bot_token=telegram_bot_token)


def send_post_to_telegram_sync(image_path: str, caption: str):
    """
    Синхронная оболочка для асинхронной отправки через Telethon,
    чтобы можно было вызвать из FastAPI-эндпоинта без async.
    """
    async def send():
        await client.send_file(telegram_chat_id, image_path, caption=caption, parse_mode='html')

    asyncio.get_event_loop().run_until_complete(send())


@app.post("/generate-post")
def generate_post_api(topic: Topic):
    """
    Быстрая синхронная версия для PyCharm запуска:
    генерируем контент, картинку, отправляем в Telegram.
    """
    content = generate_content(topic.topic)
    img = generate_image_with_stability(content["title"])
    create_image_with_blur_and_text(img, content["title"])
    caption = f"<b>{content['title']}</b>\n\n{content['meta_description']}\n\n{content['post_content']}"

    # Отправка синхронно через обертку
    send_post_to_telegram_sync("story_text.jpg", caption)

    return {"message": "Пост сгенерирован и отправлен в Telegram"}


@app.get("/")
def root():
    return {"message": "Service is running"}


@app.get("/heartbeat")
def heartbeat_api():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
