import base64
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from telethon import TelegramClient
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Получение ключей из переменных окружения
openai.api_key = os.getenv("OPENAI_API_KEY")
currentsapi_key = os.getenv("CURRENTS_API_KEY")
stability_api_key = os.getenv("STABILITY_API")  # Новый ключ для Stability AI
telegram_api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
telegram_api_hash = os.getenv("TELEGRAM_API_HASH")
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = int(os.getenv("TELEGRAM_CHAT_ID"))  # ID чата или канала для публикации
telegram_channel_username = '@testPE10_5'  # Имя канала с @

# Проверяем, что все необходимые ключи установлены
if not all([openai.api_key, currentsapi_key, stability_api_key,
            telegram_api_id, telegram_api_hash, telegram_bot_token, telegram_chat_id]):
    raise ValueError("Необходимо установить переменные окружения: OPENAI_API_KEY, CURRENTS_API_KEY, STABILITY_API, "
                     "TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")


class Topic(BaseModel):
    topic: str  # Модель для входящей темы


#def get_recent_news(topic: str) -> str:
#    """
#    Запрос последних новостей с помощью Currents API по заданной теме.
#    Возвращает объединённые заголовки первых 5 новости или сообщение о пустом результате.
#    """
#    url = "https://api.currentsapi.services/v1/latest-news"
#    params = {
#        "language": "en",
#        "keywords": topic,
#        "apiKey": currentsapi_key
#    }
#    response = requests.get(url, params=params)
#    if response.status_code != 200:
#        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {response.text}")

#    news_data = response.json().get("news", [])
#    if not news_data:
#        return "Свежих новостей не найдено."

#    return "\n".join([article["title"] for article in news_data[:5]])


def generate_content(topic: str) -> dict:
    """
    Генерирует заголовок, мета-описание и полный пост на основе темы и свежих новостей.
    Использует модель OpenAI GPT-4o-mini.
    Возвращает словарь с title, meta_description и post_content.
    """
    #recent_news = get_recent_news(topic)
    try:
        title = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Придумайте привлекательный и точный заголовок для статьи на тему '{topic}', с "
                           f"учётом актуальных новостей. Заголовок должен быть интересным и ясно "
                           f"передавать суть темы. Исключить специальные символы типа #, !, _, (, ), -, и подобные. Ответ на английском языке."
            }],
            max_tokens=20,
            temperature=0.5,
            stop=["\n"]
        ).choices[0].message.content.strip()

        meta_description = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Напишите мета-описание для статьи с заголовком: '{title}'. Оно должно быть полным, "
                           f"информативным и содержать основные ключевые слова. Исключить специальные символы типа "
                           f"#, !, _, (, ), -, и подобные. Ответ на английском языке."
            }],
            max_tokens=40,
            temperature=0.5,
            stop=["."]
        ).choices[0].message.content.strip()

        post_content = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Напишите подробную статью на тему '{topic}', используя последние новости. 
                Статья должна быть:
                1. Информативной и логичной
                2. Содержать не менее 100 символов
                3. Иметь четкую структуру с подзаголовками
                4. Включать анализ текущих трендов
                5. Иметь вступление, основную часть и заключение
                6. Включать примеры из актуальных новостей
                7. Каждый абзац должен быть не менее 3-4 предложений
                8. Текст должен быть легким для восприятия и содержательным
                9. Исключить специальные символы типа #, !, _, (, ), -, и подобные. Ответ на английском языке."""
            }],
            max_tokens=100,
            temperature=0.5,
            presence_penalty=0.6,
            frequency_penalty=0.6
        ).choices[0].message.content.strip()

        return {"title": title, "meta_description": meta_description, "post_content": post_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации контента: {str(e)}")


def generate_image_with_stability(prompt: str, width=512, height=512) -> Image.Image:
    """
    Генерация квадратного изображения с помощью Stability AI (api.stability.ai),
    используя multipart/form-data с параметрами запроса,
    возвращает PIL Image объект.
    """
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {stability_api_key}"
    }
    # В multipart-запросе тело с параметрами в data, а files содержат пустой файл согласно примеру
    data = {
        "prompt": prompt,
        "output_format": "jpeg",
    }
    files = {"none": ''}  # пустое поле для multipart, чтобы не было ошибки Content-Type

    response = requests.post(url, headers=headers, data=data, files=files)

    if response.status_code == 200:
        # В ответе напрямую содержится бинарное изображение (jpeg)
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes).convert("RGBA")
        return img
    else:
        # При ошибке - выбрасываем исключение с деталями
        try:
            error_json = response.json()
        except Exception:
            error_json = response.text
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации изображения: {error_json}")


def create_image_with_blur_and_text(img: Image.Image, title: str, output_path='story_text.jpg'):
    """
    Добавляет к сгенерированному изображению размытую нижнюю четверть с градиентом и накладывает сверху заголовок.
    Сохраняет конечное изображение без альфа канала в JPEG.
    """
    width, height = img.size

    # Нижняя четверть
    blur_height = height // 4
    bottom_part = img.crop((0, height - blur_height, width, height))

    # Размываем нижнюю часть
    blurred = bottom_part.filter(ImageFilter.GaussianBlur(radius=10))

    # Создаем маску трафарета с градиентом alpha (плавный переход прозрачности)
    gradient = Image.new('L', (width, blur_height))
    for y in range(blur_height):
        # alpha меняется от 0 (прозрачный) сверху к 255 (непрозрачный) снизу
        alpha = int(255 * (y / blur_height))
        for x in range(width):
            gradient.putpixel((x, y), alpha)

    # Устанавливаем альфа-канал в размытой части
    blurred.putalpha(gradient)

    # Накладываем размытую часть на оригинал
    img.paste(blurred, (0, height - blur_height), blurred)

    # Создаём инструмент рисования текста
    draw = ImageDraw.Draw(img)

    # Задаём шрифт с поддержкой кириллицы
    font_path = 'Roboto-VariableFont_wdth,wght.ttf'
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)

    # Разбиваем заголовок на строки по ширине с запасом
    lines = []
    words = title.split()
    line = ""
    margin = 20
    max_width = width - 2 * margin
    for word in words:
        test_line = f"{line} {word}".strip()
        if draw.textlength(test_line, font=font) <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    # Расчет высоты текста и начальной позиции для центровки по вертикали и горизонтали
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    total_text_height = line_height * len(lines)
    y_text = height - blur_height + (blur_height - total_text_height) // 2

    # Рисуем текст красным цветом и центрируем по горизонтали
    for line in lines:
        text_width = draw.textlength(line, font=font)
        x_text = (width - text_width) // 2
        draw.text((x_text, y_text), line, font=font, fill=(255, 0, 0))
        y_text += line_height

    # Сохраняем без альфа-канала в JPEG
    img.convert("RGB").save(output_path, format="JPEG")


# Инициализация клиента Telethon один раз при старте
client = TelegramClient('bot_session', telegram_api_id, telegram_api_hash).start(bot_token=telegram_bot_token)


async def send_post_to_telegram(image_path: str, caption: str):
    """
    Отправляет изображение и подпись в Telegram канал/чат через Telethon.
    """
    await client.send_file(telegram_channel_username, image_path, caption=caption, parse_mode='html')


@app.post("/generate-post")
async def generate_post_api(topic: Topic):
    """
    Основной API-метод:
    1. Генерирует контент (title, meta_description, пост) по теме
    2. Генерирует изображение через Stability AI по заголовку
    3. Добавляет размытую нижнюю четверть и накладывает заголовок
    4. Сохраняет изображение
    5. Отправляет в Telegram с подписью (meta_description + post_content)
    """
    content = generate_content(topic.topic)

    # Генерируем исходное изображение по заголовку
    img = generate_image_with_stability(content["title"])
    # Формируем итоговое изображение с текстом на размытой части
    create_image_with_blur_and_text(img, content["title"])

    # Формируем подпись с html разметкой, исправляя возможное превышение лимитов Telegram можно делать по частям (зависит от длины)
    caption = f"<b>{content['title']}</b>\n\n{content['meta_description']}\n\n{content['post_content']}"

    # Отправляем картинку и подпись в Telegram
    await send_post_to_telegram("story_text.jpg", caption)

    return {"message": "Пост сгенерирован и отправлен в Telegram"}


@app.get("/")
async def root():
    return {"message": "Service is running"}


@app.get("/heartbeat")
async def heartbeat_api():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    import asyncio

    # Функция для последовательного запуска генерации и отправки по теме из терминала
    def run_with_input():
        topic = input("Введите тему для генерации поста: ").strip()
        print(f"Генерируем пост для темы: {topic}")
        content = generate_content(topic)
        img = generate_image_with_stability(content["title"])
        create_image_with_blur_and_text(img, content["title"])
        caption = f"<b>{content['title']}</b>\n\n{content['meta_description']}\n\n{content['post_content']}"

        # Асинхронно отправляем в Telegram
        asyncio.get_event_loop().run_until_complete(send_post_to_telegram("story_text.jpg", caption))
        print("Пост сгенерирован и отправлен в Telegram.")

    # Запускаем сервер и одновременно спрашиваем тему в терминале
    # Можно вызвать по желанию run_with_input(), либо запустить сервер
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_with_input()
    else:
        port = int(os.getenv("PORT", 8000))
        uvicorn.run("app:app", host="0.0.0.0", port=port)

