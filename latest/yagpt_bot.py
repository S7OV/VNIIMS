import os
import logging
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from embedding_qa import run_embedding_qa  # Импортируем функцию из embedding_qa.py
from embedding_bd import run_embedding_bd  # Импортируем функцию из embedding_bd.py

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Подгружаем переменные окружения
load_dotenv()

# Передаем секретные данные в переменные
TOKEN = os.environ.get("TG_TOKEN")

# Функция для общения с chatgpt
def get_answer(text):
    payload = {"text": text}
    response = requests.post("http://127.0.0.1:5000/api/get_answer", json=payload)
    res = response.json()
    return res

# Функция для разбивки текста на части
def split_text(text, max_length=4096):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# Функция-обработчик текстовых сообщений
async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.message.chat_id, text='Ваш запрос обрабатывается...')
    res = get_answer(update.message.text)
    messages = split_text(res['message'])
    for message in messages:
        await context.bot.send_message(chat_id=update.message.chat_id, text=message)

# Обработчик команды /embedding_qa
async def embedding_qa(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.message.chat_id, text='Производится обработка базы вопросов-ответов...')
    result = run_embedding_qa()
    await context.bot.send_message(chat_id=update.message.chat_id, text=result)

# Обработчик команды /embedding_bd
async def embedding_bd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.message.chat_id, text='Производится обработка базы знаний...')
    result = run_embedding_bd()
    await context.bot.send_message(chat_id=update.message.chat_id, text=result)    

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = ('Здравствуйте! Я прототип нейросетевого эксперта Всероссийского научно-исследовательского института '
                       'метрологической службы. Готов ответить на Ваши вопросы в области обеспечения единства измерений.')
    await context.bot.send_message(chat_id=update.message.chat_id, text=welcome_message)

# Обработчик ошибок
def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

async def error_handler(update, context):
    logging.error(msg="Exception while handling an update:", exc_info=context.error)
    # Дополнительная логика обработки ошибок

def main():
    # Создаем приложение и передаем в него токен бота
    application = Application.builder().token(TOKEN).build()
    print('Бот запущен...')

    # Добавление обработчиков
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text))
    application.add_handler(CommandHandler("embedding_qa", embedding_qa))  # Добавляем обработчик команды /embedding_qa
    application.add_handler(CommandHandler("embedding_bd", embedding_bd))  # Добавляем обработчик команды /embedding_bd
    application.add_handler(CommandHandler("start", start))  # Добавляем обработчик команды /start
    # Обработчик ошибок
    application.add_error_handler(error_handler)

    # Запуск бота (нажать Ctrl+C для остановки)
    application.run_polling()
    print('Бот остановлен')

if __name__ == "__main__":
    main()
