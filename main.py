import telebot
import requests
import time
from io import BytesIO
from PIL import Image
import json
import base64

# Токен вашего бота
TOKEN = '.'
# API ключи для FusionBrain
FUSIONBRAIN_API_KEY = '.'
FUSIONBRAIN_SECRET_KEY = '.'

class FusionBrainAPI:
    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_pipeline(self):
        try:
            response = requests.get(self.URL + 'key/api/v1/pipelines', headers=self.AUTH_HEADERS)
            response.raise_for_status()
            data = response.json()
            return data[0]['id']
        except Exception as e:
            print(f"Ошибка получения pipeline: {e}")
            return None

    def generate(self, prompt, pipeline_id, images=1, width=512, height=512):
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "style": "CLASSICISM",
            "generateParams": {
                "query": f"Text: '{prompt}' on a classicism-style background"
            }
        }
        data = {
            'pipeline_id': (None, pipeline_id),
            'params': (None, json.dumps(params), 'application/json')
        }
        try:
            response = requests.post(self.URL + 'key/api/v1/pipeline/run', headers=self.AUTH_HEADERS, files=data)
            response.raise_for_status()
            return response.json()['uuid']
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return None

    def check_generation(self, request_id, attempts=20, delay=5):
        while attempts > 0:
            try:
                response = requests.get(self.URL + 'key/api/v1/pipeline/status/' + request_id, headers=self.AUTH_HEADERS)
                response.raise_for_status()
                data = response.json()
                if data['status'] == 'DONE':
                    return data['result']['files']
                elif data['status'] in ['FAILED', 'REJECTED']:
                    return None
                attempts -= 1
                time.sleep(delay)
            except Exception as e:
                print(f"Ошибка проверки статуса: {e}")
                return None
        return None

bot = telebot.TeleBot(TOKEN)
api = FusionBrainAPI('https://api-key.fusionbrain.ai/', FUSIONBRAIN_API_KEY, FUSIONBRAIN_SECRET_KEY)
pipeline_id = api.get_pipeline()

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Напиши текст, который хочешь превратить в изображение в стиле классицизм.")

# Обработчик текстовых сообщений
@bot.message_handler(content_types=['text'])
def generate_image(message):
    prompt = message.text
    chat_id = message.chat.id
    bot.send_message(chat_id, "Генерирую изображение с текстом в стиле классицизм, пожалуйста, подождите...")

    try:
        if not pipeline_id:
            bot.send_message(chat_id, "Ошибка: не удалось инициализировать pipeline.")
            return

        # Запрос на генерацию изображения
        task_uuid = api.generate(prompt, pipeline_id)
        if not task_uuid:
            bot.send_message(chat_id, "Ошибка: не удалось создать задачу генерации.")
            return

        # Проверка статуса генерации
        files = api.check_generation(task_uuid)
        if files and len(files) > 0:
            # Предполагаем, что files[0] содержит base64-строку изображения
            base64_string = files[0]
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]  # Удаляем префикс data:image
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            bot.send_photo(chat_id, image_buffer)
        else:
            bot.send_message(chat_id, "Ошибка: не удалось получить изображение.")

    except Exception as e:
        error_msg = str(e)[:1000]  # Ограничиваем длину сообщения об ошибке
        bot.send_message(chat_id, f"Произошла ошибка: {error_msg}")

# Запуск бота
if __name__ == "__main__":
    while True:
        try:
            bot.polling(none_stop=True, timeout=60, long_polling_timeout=30)
        except Exception as e:
            print(f"Ошибка polling: {e}")
            time.sleep(15)
