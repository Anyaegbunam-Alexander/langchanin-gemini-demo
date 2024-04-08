import os

from astrapy.db import AstraDB
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASTRA_DB_SECURE_BUNDLE_PATH = os.environ.get("ASTRA_DB_SECURE_BUNDLE_PATH")
ASTRA_DB_NAMESPACE = os.environ.get("ASTRA_DB_NAMESPACE")
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# Initialize the client
db = AstraDB(
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

session = Cluster(
    cloud={"secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH},
    auth_provider=PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN),
).connect()


message_history = CassandraChatMessageHistory(
    session_id="choose-adventure-session-id",
    session=session,
    keyspace=ASTRA_DB_NAMESPACE,
    ttl_seconds=3600,
)

message_history.clear()

cass_buff_memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

template = """
You are now the guide of a mystical journey in the Whispering Woods.
A traveler named Aldona seeks the lost Amulet of Uden.
You must navigate him through a lot of challenges, choices, and consequences, fights, battles,
dynamically adapting the tale based on the traveler's decisions.
Your goal is to create a long branching narrative experience where each choice
leads to a new path, ultimately determining Aldona's fate.

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a lot of paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that
explains the death and ends in the text: "The End.", I will search for this text to end the game
4. Let there be action, drama, friendship, romance, and betrayal.
Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=cass_buff_memory)

reply = "start the game"

while True:
    response = llm_chain.predict(human_input=reply)
    print(response.strip())

    if "The End" in response:
        break

    reply = input("Your reply: ")
