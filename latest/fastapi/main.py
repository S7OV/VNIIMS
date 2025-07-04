from fastapi import FastAPI
from answers import Answer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# инициализация индексной базы

answers = Answer()

# класс с типами данных параметров 
class Item(BaseModel): 
    text: str

# создаем объект приложения
app = FastAPI()

# настройки для работы запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# функция обработки get запроса + декоратор 
@app.get("/")
def read_root():
    return {"message": "answer"}

# функция обработки post запроса + декоратор 
@app.post("/api/get_answer")
def get_answer(question: Item):
    answer = answers.get_answer(query=question.text)
    return {"message": answer}

# асинхронная функция обработки post запроса + декоратор 
@app.post("/api/get_answer_async")
async def get_answer_async(question: Item):
    answer = await answers.async_get_answer(query=question.text)
    return {"message": answer}

