import boto3
import io
from io import BytesIO
import pandas as pd
import os
from langchain.docstore.document import Document as Doc
import requests
import tiktoken
import datetime
from datetime import datetime
import numpy as np
import pickle
import re
import time
from docx import Document
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml
from xml.etree.ElementTree import fromstring as parse_xml
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, CharacterTextSplitter



def run_embedding_bd():

    # Передаем секретные данные в переменные
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
    bucket_name = 'vniims-bd'
    # Получаем список объектов в бакете
    response = s3.list_objects_v2(Bucket=bucket_name)

    def preprocess_res_dub(document):
        markdown_text = ''
        markdown_lines = []  # список для хранения строк в формате Markdown
        current_paragraph = ""  # текущий параграф
        current_header_level = 1  # текущий уровень заголовка
        count_styles = 0  # счетчик стилей
        reset_flag = False  # флаг для сброса

        # проходим по каждому параграфу в документе
        for i, paragraph in enumerate(document.paragraphs):
            markdown_text = paragraph.text  # получаем текст параграфа

            # если текст параграфа не пустой
            if markdown_text:
                p_xml = paragraph._p.xml  # получаем xml параграфа
                root = parse_xml(r'{}'.format(p_xml))  # парсим xml
                namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}  # пространство имен для xml
                style = root.find('.//w:pPr/w:pStyle', namespaces=namespaces)  # ищем стиль параграфа
                outline_lvl = root.find('.//w:pPr/w:outlineLvl', namespaces=namespaces)  # ищем уровень заголовка

                # если уровень заголовка определен
                if outline_lvl is not None:
                    outline_lvl_val = int(outline_lvl.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val'))  # получаем значение уровня заголовка
                    if current_header_level != outline_lvl_val + 1:
                        count_styles = 0  # если текущий уровень заголовка не равен полученному, сбрасываем счетчик стилей
                    current_header_level = outline_lvl_val + 1  # обновляем текущий уровень заголовка

                    # если стиль параграфа определен
                    if style is not None:
                        style_val = style.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')  # получаем значение стиля
                        if style_val == 'ConsPlusTitle':  # если стиль равен 'ConsPlusTitle'
                            reset_flag = False  # сбрасываем флаг
                            count_styles += 1  # увеличиваем счетчик стилей
                            if count_styles == 1:
                                current_paragraph = "#" * current_header_level + " " + markdown_text  # если счетчик стилей равен 1, формируем заголовок
                            elif count_styles > 1:  # если счетчик стилей больше 1
                                current_paragraph = current_paragraph + " " + markdown_text  # добавляем текст к текущему параграфу
                        else:  # если стиль не равен 'ConsPlusTitle'
                            count_styles = 0  # сбрасываем счетчик стилей
                            if not reset_flag:  # если флаг не установлен
                                markdown_lines.append(current_paragraph + "\n")  # добавляем текущий параграф в список строк
                                markdown_lines.append(current_paragraph + "\n")  # добавляем текущий параграф в список строк еще раз
                                reset_flag = True  # устанавливаем флаг
                            markdown_lines.append(markdown_text + "\n")  # добавляем текст параграфа в список строк

                else:  # если уровень заголовка не определен
                    if style is not None:
                        style_val = style.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')  # получаем значение стиля
                        if style_val == 'ConsPlusTitle':  # если стиль равен 'ConsPlusTitle'
                            reset_flag = False  # сбрасываем флаг
                            count_styles += 1  # увеличиваем счетчик стилей
                            if count_styles == 1:
                                current_paragraph = "#" * current_header_level + " " + markdown_text  # если счетчик стилей равен 1, формируем заголовок
                            elif count_styles > 1:  # если счетчик стилей больше 1
                                current_paragraph = current_paragraph + " " + markdown_text  # добавляем текст к текущему параграфу
                        else:  # если стиль не равен 'ConsPlusTitle'
                            count_styles = 0  # сбрасываем счетчик стилей
                            if not reset_flag:  # если флаг не установлен
                                markdown_lines.append(current_paragraph + "\n")  # добавляем текущий параграф в список строк
                                markdown_lines.append(current_paragraph + "\n")  # добавляем текущий параграф в список строк еще раз
                                reset_flag = True  # устанавливаем флаг
                            match = re.match(r'^\d+\.\w+', markdown_text)    # поиск пунктов начианющихся с цифры и до пробела
                            if match:
                                markdown_text = '\n' + '#' * (current_header_level + 1) + ' пункт ' + match.group() + '\n' + markdown_text
                                markdown_lines.append(markdown_text + "\n")  # добавляем новый текст параграфа в список строк
                            else:
                                markdown_lines.append(markdown_text + "\n")  # добавляем текст параграфа в список строк

        return markdown_lines  # возвращаем список строк в формате Markdown

    # функция вызова предобработки документов
    def pre_doc_markdown(response, doc_markdown):
        for obj in response['Contents']:
            if obj['Key'].startswith('bd/') and obj['Key'].endswith('.docx'):
                docx_file_key = obj['Key']
                print(f"Найден файл: {docx_file_key}")
                # Загружаем файл из S3
                response = s3.get_object(Bucket=bucket_name, Key=docx_file_key)
                file_content = response['Body'].read()
                document = Document(io.BytesIO(file_content))
                print(f"Тип документа: {type(document)}")
                doc_markdown += ''.join(preprocess_res_dub(document))  # предобработка текста
        return doc_markdown
    
    doc_markdown = ''
    get_pre_doc_markdown = False
    doc_markdown_get = ''
    # Проверяем, есть ли объекты в бакете и актуальность объектов
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].startswith('bd/') and obj['Key'].endswith('.docx'):
                docx_file_key = obj['Key']
                if docx_file_key:
                    # Проверяем метаданные файла
                    head_response = s3.head_object(Bucket=bucket_name, Key=docx_file_key)
                    if 'Metadata' in head_response and head_response['Metadata'].get('Processed') != 'true':
                        get_pre_doc_markdown = True
                        print("Необходимо обновить базу знаний.")
                  
                    else:
                        print (f"Документ {docx_file_key} не обновлен")
        if get_pre_doc_markdown:
            print("Создание документа в формате markdown.")
            doc_markdown_get = pre_doc_markdown(response, doc_markdown)
        else: 
          result = "База не требует обновления" 
          return result
    else:
        print("Нет объектов в бакете.")

    # Сохраняем обработанные файлы на локальной машине
    local_file_name_markdown = 'v4.5_doc_markdown_vniims.txt'
    with open(local_file_name_markdown, 'a', encoding='utf-8') as f:  # сохраним в файл
        f.write(doc_markdown_get)

    # Указываем имя файла в бакете
    s3_file_name_markdown = local_file_name_markdown

    # Загружаем файл markdown в бакет S3
    s3.upload_file(local_file_name_markdown, bucket_name, s3_file_name_markdown)



    def num_tokens_from_string(string: str, encoding_name: str) -> int:
          """Возвращает количество токенов в строке"""
          encoding = tiktoken.get_encoding(encoding_name)
          num_tokens = len(encoding.encode(string))
          return num_tokens

    def split_text(text, max_count):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
        fragments = markdown_splitter.split_text(text)

        # Подсчет токенов для каждого фрагмента
        fragment_token_counts = [num_tokens_from_string(fragment.page_content, "cl100k_base") for fragment in fragments]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_count,
            chunk_overlap=0,
            length_function=lambda x: num_tokens_from_string(x, "cl100k_base")
        )

        source_chunks = [
            Doc(page_content=chunk, metadata=fragment.metadata)
            for fragment in fragments
            for chunk in splitter.split_text(fragment.page_content)
        ]

        # Подсчет токенов для каждого source_chunk
        source_chunk_token_counts = [num_tokens_from_string(chunk.page_content, "cl100k_base") for chunk in source_chunks]

        return source_chunks, fragments

    source_chunks, fragments = split_text(doc_markdown_get, 500)
    print("Общее количество чанков: ",len(source_chunks))
    print("Первый чанк ", source_chunks[0].page_content)
    print("Metadata Первого чанка", source_chunks[0].metadata)
    print("Крайний чанк ", source_chunks[len(source_chunks)-1])
    print("Metadata крайнего чанка", source_chunks[len(source_chunks)-1].metadata)

    ID_FOLDER = 'b1gmbmom8qe6nho8em9o'
    OAuth_token = 'y0_AgAAAABw6jr2AATuwQAAAAEGWX2tAADeOEZ3lbpDC5ChJOOggtropfOo4Q'

    # URL для получения токена
    URL = "https://iam.api.cloud.yandex.net/iam/v1/tokens"

    # Функция для получения IAM-токена
    def get_iam_token(oauth_token):
        headers = {"Content-Type": "application/json"}
        data = {
            "yandexPassportOauthToken": oauth_token
        }
        response = requests.post(URL, headers=headers, json=data)
        response_data = response.json()
        return response_data["iamToken"], response_data["expiresAt"]

    IAM_TOKEN, expiresAt = get_iam_token(OAuth_token)

    doc_uri = f"emb://{ID_FOLDER}/text-search-doc/latest"
    query_uri = f"emb://{ID_FOLDER}/text-search-query/latest"

    embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {IAM_TOKEN}", "x-folder-id": f"{ID_FOLDER}"}

    def get_embedding(text: str, text_type: str = "doc") -> np.array:
        query_data = {
            "modelUri": doc_uri if text_type == "doc" else query_uri,
            "text": text,
        }

        return np.array(
            requests.post(embed_url, json=query_data, headers=headers).json()["embedding"]
        )

    #Переведем текст который необходим для Embedding YandexGPT
    doc_texts = []
    for i in range(len(source_chunks)):
        text = f'{source_chunks[i].page_content}'
        doc_texts.append(text)

    #Переведем все метадата чанков в вид, который необходим для Embedding YandexGPT
    doc_metadata = []
    separator = ", "
    for i in range(len(source_chunks)):
        result = separator.join([f"{v}" for k, v in source_chunks[i].metadata.items()])
        text = f'{result},'
        doc_metadata.append(text)

    docs_embedding = []
    for i in range(len(source_chunks)):
        text = f'{source_chunks[i].page_content}'
        embd = get_embedding(text)
        time.sleep(0.1)        
        docs_embedding.append(embd)
        print ('Отработан чанк № ',i)

    docs_embedding_arr = np.array(docs_embedding)

    # Сохраняем массив в файл .npy на локальной машине
    local_file_name_bd = 'embedding_bd_ya_01-500.npy'
    np.save(local_file_name_bd, docs_embedding_arr)

    # Указываем имя файла в бакете
    s3_file_name_bd = local_file_name_bd

    # Загружаем файл .npy в бакет S3
    s3.upload_file(local_file_name_bd, bucket_name, s3_file_name_bd)

    # Удаляем локальный файл после загрузки (опционально)
    #os.remove(local_file_name_emq)

    print(f"Файл {s3_file_name_bd} успешно загружен в бакет {bucket_name}.") 

    # Сохраняем метаданные базы данных в файл pkl на локальной машине
    local_file_name_md = 'doc_metadata_ya_01-500.pkl'

    with open(local_file_name_md, 'wb') as file:
        pickle.dump(doc_metadata, file)

    # Указываем имя файла в бакете
    s3_file_name_md = local_file_name_md

    # Загружаем файл .npy в бакет S3
    s3.upload_file(local_file_name_md, bucket_name, s3_file_name_md)

    # Удаляем локальный файл после загрузки (опционально)
    os.remove(local_file_name_md)

    print(f"Файл {s3_file_name_md} успешно загружен в бакет {bucket_name}.")

    # Сохраняем вопросы в файл pkl на локальной машине
    local_file_name_doc = 'doc_embd_ya_01-500.pkl'

    with open(local_file_name_doc, 'wb') as file:
        pickle.dump(doc_texts, file)

    # Указываем имя файла в бакете
    s3_file_name_doc = local_file_name_doc

    # Загружаем файл .npy в бакет S3
    s3.upload_file(local_file_name_doc, bucket_name, s3_file_name_doc)

    # Удаляем локальный файл после загрузки (опционально)
    os.remove(local_file_name_doc)

    print(f"Файл {s3_file_name_doc} успешно загружен в бакет {bucket_name}.")

    result = f"Обработка базы знаний произведена!\nФайлы {s3_file_name_bd},{s3_file_name_md},{s3_file_name_doc} успешно загружены в бакет {bucket_name}."

    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].startswith('bd/') and obj['Key'].endswith('.docx'):
                docx_file_key = obj['Key']
                print(f"Найден файл: {docx_file_key}")
                # Обновляем метаданные файла, чтобы отметить его как обработанный -!!!Нужно засунуть это в конец!!!
                s3.copy_object(
                    Bucket=bucket_name,
                    CopySource={'Bucket': bucket_name, 'Key': docx_file_key},
                    Key=docx_file_key,
                    Metadata={'Processed': 'true'},
                    MetadataDirective='REPLACE'
                )
                print(f"Файл: {docx_file_key} отмечен")    

    return result