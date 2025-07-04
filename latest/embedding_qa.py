import boto3
import io
from io import BytesIO
import pandas as pd
import os
from langchain.docstore.document import Document
import requests
import datetime
from datetime import datetime
import numpy as np
import pickle
import time
from dotenv import load_dotenv

# Подгружаем переменные окружения
load_dotenv()

def run_embedding_qa():
    access_key = os.environ.get("access_key")
    secret_key = os.environ.get("secret_key")
    # Создаем сессию с использованием учетных данных
    session = boto3.session.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    # Создаем клиента для работы с S3
    s3 = session.client(service_name='s3', endpoint_url='https://storage.yandexcloud.net')

    # Указываем имя бакета
    bucket_name = 'vniims-qa'
    # Получаем список объектов в бакете
    response = s3.list_objects_v2(Bucket=bucket_name)

    # Ищем файл с расширением .xlsx
    xlsx_file_key = None
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.xlsx'):
                xlsx_file_key = obj['Key']
                print ('Найден файл ', xlsx_file_key)
                break

    if xlsx_file_key:
        # Проверяем метаданные файла
        head_response = s3.head_object(Bucket=bucket_name, Key=xlsx_file_key)
        if 'Metadata' in head_response and head_response['Metadata'].get('Processed') == 'true':
            result = "Файл уже был обработан."
            print(result)
            return result

        # Загружаем файл из S3
        response = s3.get_object(Bucket=bucket_name, Key=xlsx_file_key)
        file_content = response['Body'].read()

        # Читаем файл в датафрейм
        df = pd.read_excel(io.BytesIO(file_content))
        print ('Файл загружен ', type(io.BytesIO(file_content)))

    else:
        print("Файл с расширением .xlsx не найден в бакете.")
        return "Файл с расширением .xlsx не найден в бакете."

    # Функция для создания объекта Document из строки DataFrame
    def create_document(row):
        return Document(
            page_content=row['Вопрос'],
            metadata={'Ответ': row['Ответ']}
        )

    # Применяем функцию ко всем строкам DataFrame
    documents = df.apply(create_document, axis=1).tolist()

    # Переведем все чанки в вид, который необходим для Embedding YandexGPT
    doc_texts = []
    for i in range(len(documents)):
        text = f'{documents[i].page_content},'
        doc_texts.append(text)

    def get_embedding(text: str, text_type: str = "doc") -> np.array:
        ID_FOLDER = os.environ.get("ID_FOLDER")
        OAuth_token = os.environ.get("OAuth_token")
        doc_uri = f"emb://{ID_FOLDER}/text-search-doc/latest"
        query_uri = f"emb://{ID_FOLDER}/text-search-query/latest"
        embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
                # Получение IAM-токена (с помощью request)
                                # URL для получения токена
        URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
        headers = {"Content-Type": "application/json"}
        data = {
            "yandexPassportOauthToken": OAuth_token
        }
        response = requests.post(URL, headers=headers, json=data)
        IAM_TOKEN = response.json()["iamToken"]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {IAM_TOKEN}", "x-folder-id": f"{ID_FOLDER}"}


        
        query_data = {
            "modelUri": doc_uri if text_type == "doc" else query_uri,
            "text": text,
        }
        response = requests.post(embed_url, json=query_data, headers=headers)
        
        # Проверка статуса ответа
        if response.status_code != 200:
            print(f"Ошибка при запросе эмбеддинга: {response.status_code} - {response.text}")
            return None
        
        embd = response.json()
        
        # Проверка наличия ключа 'embedding' в ответе
        if 'embedding' not in embd:
            print(f"Ключ 'embedding' отсутствует в ответе: {embd}")
            return None
        
        embd_json = embd['embedding']
        #print(embd)

        result = np.array(embd_json)
        print('Эмбеддинг получен', type(result))
        return result

    # Embedding базы ответов
    docs_embedding = []
    for doc_text in doc_texts:
        embedding = get_embedding(doc_text)
        if embedding is not None:
            docs_embedding.append(embedding)
            time.sleep(0.1)
        else:
            print(f"Не удалось получить эмбеддинг для текста: {doc_text}")

    docs_embedding_arr = np.array(docs_embedding)

    # Сохраняем массив в файл .npy на локальной машине
    local_file_name_emq = 'question_embedding_05-117.npy'
    np.save(local_file_name_emq, docs_embedding_arr)

    # Указываем имя файла в бакете
    s3_file_name_emq = local_file_name_emq

    # Загружаем файл .npy в бакет S3
    s3.upload_file(local_file_name_emq, bucket_name, s3_file_name_emq)

    # Удаляем локальный файл после загрузки (опционально)
    # os.remove(local_file_name_emq)

    print(f"Файл {s3_file_name_emq} успешно загружен в бакет {bucket_name}.")

    # Переведем все метадата ответов в вид, который необходим для Embedding YandexGPT
    answer_metadata = []
    for i in range(len(documents)):
        text = f'{documents[i].metadata}'
        answer_metadata.append(text)

    # Сохраняем ответы в файл pkl на локальной машине
    local_file_name = 'answer_metadata_05-117.pkl'

    with open(local_file_name, 'wb') as file:
        pickle.dump(answer_metadata, file)

    # Указываем имя файла в бакете
    s3_file_name = local_file_name

    # Загружаем файл .npy в бакет S3
    s3.upload_file(local_file_name, bucket_name, s3_file_name)

    # Удаляем локальный файл после загрузки (опционально)
    os.remove(local_file_name)

    print(f"Файл {s3_file_name} успешно загружен в бакет {bucket_name}.")

    # Сохраняем вопросы в файл pkl на локальной машине
    local_file_name_q = 'question_texts_05-117.pkl'

    with open(local_file_name_q, 'wb') as file:
        pickle.dump(doc_texts, file)

    # Указываем имя файла в бакете
    s3_file_name_q = local_file_name_q

    # Загружаем файл .npy в бакет S3
    s3.upload_file(local_file_name_q, bucket_name, s3_file_name_q)

    # Удаляем локальный файл после загрузки (опционально)
    os.remove(local_file_name_q)

    print(f"Файл {s3_file_name_q} успешно загружен в бакет {bucket_name}.")

    # Обновляем метаданные файла, чтобы отметить его как обработанный
    s3.copy_object(
        Bucket=bucket_name,
        CopySource={'Bucket': bucket_name, 'Key': xlsx_file_key},
        Key=xlsx_file_key,
        Metadata={'Processed': 'true'},
        MetadataDirective='REPLACE'
    )

    result = f"Обработка базы вопрос-ответ произведена!\nФайлы {s3_file_name},{s3_file_name_q},{s3_file_name_emq} успешно загружены в бакет {bucket_name}."

    return result
