from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, TextLoader

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from fastapi import FastAPI
from pydantic import BaseModel


class Message(BaseModel):
    user_id: str
    message: str


loader = DirectoryLoader('./Data', glob="preproc*.txt")

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n", ". ", "."],
    keep_separator=False,
    add_start_index=True,
    chunk_size=256,
    chunk_overlap=64,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(loader.load())
embeddings_model = SentenceTransformerEmbeddings(model_name='cointegrated/rubert-tiny2')
db = Chroma.from_documents(chunks, embeddings_model, persist_directory="./chroma_db_preproc_docs")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path='./llama-2-7b-chat.Q4_K_M.gguf',
    temperature=0.90,
    max_tokens=1000,
    top_p=0.75,
    callback_manager=callback_manager,
    n_ctx=1024,
    verbose=True,
)

app = FastAPI()
redis_backed_dict = dict()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template("Вы - отзывчивый и честный сотрудник банка Тинькофф. Люди спрашивают вас о нашем банке, и вы должны давать им правильные ответы на русском языке. У вас есть актуальная информация о нашем банке, и ваши ответы всегда должны основываться на этой информации. Пожалуйста, убедитесь, что ваш ответ максимально краток и верен. Всегда отвечайте максимально полезно. Если вы не знаете ответа на вопрос, пожалуйста, не делитесь ложной информацией. Ваши ответы должны быть составлены на граммотном русском языке. За ответы на любом другом языке, отличном от русского, вас будет ожидать серьезное наказание. Актуальная информация о нашем банке:\n{information}"),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ],
)


@app.post("/message")
def message(full_message: Message):
    user_id, message = full_message.user_id, full_message.message
    memory = redis_backed_dict.get(
        user_id,
        ConversationBufferMemory(memory_key="chat_history", input_key="question", ai_prefix="Bank",
                                 human_prefix="Human", return_messages=True)
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    relevant_info = db.similarity_search(message, k=5)
    relevant_info_prompt = '\n'.join([f'{i + 1}) {relevant_info[i].page_content}' for i in range(len(relevant_info))])
    ai_message = conversation({"information": relevant_info_prompt, "question": message})
    return {"message": ai_message["text"]}
