import os
import io
import pickle
import numpy as np
import json
import scipy
from scipy.spatial.distance import cdist
import re
import requests
import time
from dotenv import load_dotenv


class Answer:

    def __init__(self):
        # Выгрузим массива embedding вопросов для из стороннего источника
        response = requests.get('https://storage.yandexcloud.net/vniims-qa/question_embedding_04-117.npy')
        file = io.BytesIO(response.content)
        self.question_embedding = np.load(file)

        # Выгрузим подготовленную базу ответов для embedding из файла
        response = requests.get('https://storage.yandexcloud.net/vniims-qa/answer_metadata_05-117.pkl')
        file = io.BytesIO(response.content)
        self.answer_metadata = pickle.load(file)

        # Выгрузим подготовленную базу вопросов для embedding из файла
        response = requests.get('https://storage.yandexcloud.net/vniims-qa/question_texts_05-117.pkl')
        file = io.BytesIO(response.content)
        self.question_texts = pickle.load(file)

        # Выгрузим подготовленную базу для embedding из из стороннего источника
        response = requests.get('https://storage.yandexcloud.net/vniims-bd/doc_embd_ya_01-500.pkl')
        file = io.BytesIO(response.content)
        self.doc_texts = pickle.load(file)

        # загрузка массива embeddings из стороннего источника
        response = requests.get('https://storage.yandexcloud.net/vniims-bd/embedding_bd_ya_01-500.npy')
        file = io.BytesIO(response.content)
        self.docs_embedding = np.load(file)

        # Выгрузим подготовленную базу metadata для embedding из файла стороннего источника
        response = requests.get('https://storage.yandexcloud.net/vniims-bd/doc_metadata_ya_01-500.pkl')
        file = io.BytesIO(response.content)
        self.doc_metadata = pickle.load(file)

        # Получение системного промта
        response = requests.get('https://storage.yandexcloud.net/vniims-qa/system.txt')
        #response = requests.get('https://drive.google.com/uc?export=download&id=1dSpU9VMXxyO20HFsQuoHuybqP8PLQstV')
        response.raise_for_status()
        response.encoding = 'utf-8'
        self.system = response.text

        response = requests.get('https://storage.yandexcloud.net/vniims-qa/system_answer.txt')
        response.raise_for_status()
        response.encoding = 'utf-8'
        self.system_answer = response.text

        response = requests.get('https://storage.yandexcloud.net/vniims-qa/system_only.txt')
        response.raise_for_status()
        response.encoding = 'utf-8'
        self.system_only = response.text

        response = requests.get('https://storage.yandexcloud.net/vniims-qa/system_doc.txt')
        response.raise_for_status()
        response.encoding = 'utf-8'
        self.system_doc = response.text

        # получим переменные окружения из .env
        load_dotenv()

        self.ID_FOLDER = os.environ.get("ID_FOLDER")
        self.OAuth_token = os.environ.get("OAuth_token")

    def get_embedding(self, text: str, text_type: str = "doc") -> np.array:
        # URL для получения токена
        URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"

        # Получение IAM-токена (с помощью request)
        headers = {"Content-Type": "application/json"}

        data = {
            "yandexPassportOauthToken": self.OAuth_token
        }

        response = requests.post(URL, headers=headers, json=data)

        IAM_TOKEN = response.json()["iamToken"]


        doc_uri = f"emb://{self.ID_FOLDER}/text-search-doc/latest"
        query_uri = f"emb://{self.ID_FOLDER}/text-search-query/latest"

        embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {IAM_TOKEN}", "x-folder-id": f"{self.ID_FOLDER}"}
        query_data = {
            "modelUri": doc_uri if text_type == "doc" else query_uri,
            "text": text,
        }

        return np.array(
            requests.post(embed_url, json=query_data, headers=headers).json()["embedding"]
        )

    # функция вывода k релевантных чанков по косинусному расстоянию
    def get_k_max_indices(self, arr, k):
        indices = np.argpartition(arr, -k)[-k:]
        return indices

    # Функция сравнения и формирования вопроса Embedding базы данных и базы вопросов
    def contrast_embedding(self, query_text):
        # получем embedding вопроса
        query_embedding = self.get_embedding(query_text, text_type="query")
        # Вычисляем косинусное расстояние
        dist_q = cdist(query_embedding[None, :], self.question_embedding, metric="cosine")
        # Вычисляем косинусное сходство
        sim_q = 1 - dist_q
        print('Релевантность базы вопросов', np.max(sim_q))
        # ind_k = chunks_indices_db(query_text)
        dist = cdist(query_embedding[None, :], self.docs_embedding, metric="cosine")
        # Вычисляем косинусное сходство
        sim = 1 - dist
        k = 4  # количество релевантных чанков
        indices = self.get_k_max_indices(sim, k)
        # находим индексы k релевантных чанков
        ind_k = indices[0][-k:]
        print(ind_k)
        print('Релевантность базы данных', np.max(sim[0, ind_k]))
        # Подтягиваем релевантные чанки из базы данных
        docs = []
        doc_meta = []
        for ind in ind_k:
            docs.append(self.doc_texts[ind])
            doc_meta.append(self.doc_metadata[ind])
            message_content_doc = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#Отрывок документа №{i+1}:' + doc_meta[i] + ', ' + doc + '\n' for i, doc in enumerate(docs)]))
            if np.max(sim_q) < 0.4 and np.max(sim[0, ind_k]) < 0.4:
                question = f"#*Вопрос пользователя:* \n{query_text}"
                system = self.system_only
                print (question, system)
            elif np.max(sim_q) > 0.4 and np.max(sim[0, ind_k]) > 0.4 and np.max(sim_q) - np.max(sim[0, ind_k]) > 0.1:
                #Выполняем обращение к LLM с базой вопросов
                message_content = self.answer_metadata[np.argmax(sim_q)]
                #print (message_content)
                question = f"#*Вопрос пользователя:* \n{query_text}\n#*Образец вопроса пользователя:*{self.question_texts[np.argmax(sim_q)]}\n#*Образец ответа:* {message_content}"
                system = self.system_answer
                print (question, system)       
            else:
                #print (message_content)
                question = f"#*Вопрос пользователя:* \n{query_text}\n#*Отрывки нормативных документов:* {message_content_doc}"
                system = self.system_doc
                print (question, system)
            return question, system

    # Функция обращения к модели YandexGPT Pro
    def get_gpt_ya_response(self, question):
        self.question, self.system = self.contrast_embedding (question)
        # URL для получения токена
        URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
        # Получение IAM-токена (с помощью request)
        headers = {"Content-Type": "application/json"}
        data = {
            "yandexPassportOauthToken": self.OAuth_token
        }
        response = requests.post(URL, headers=headers, json=data)
        IAM_TOKEN = response.json()["iamToken"]

        prompt = {
            "modelUri": f"gpt://{self.ID_FOLDER}/yandexgpt",  # yandexgpt - модель YandexGPT pro
            "completionOptions": {
                "stream": False,
                "temperature": 0.0,
                "maxTokens": 2000
            },
            "messages": [
                {
                    "role": "system",
                    "text": f"{self.system}"
                },
                {
                    "role": "user",
                    "text": f"{self.question}"
                }
            ]
        }
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {IAM_TOKEN}"
        }
        response = requests.post(url, headers=headers, json=prompt)
        json_data = json.loads(response.text)
        answer = json_data['result']['alternatives'][0]['message']['text']
        return answer

    def get_answer(self, query: str = None):
        question = query
        gpt_answer_output = self.get_gpt_ya_response(question)
        return gpt_answer_output

    async def get_answer_async(self, query: str = None):
        question = query
        gpt_answer_output = await self.get_gpt_ya_response(question)
        return gpt_answer_output