import os, shutil

import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb.api
from langchain.memory import ConversationBufferMemory





from DocLoaders import DocLoaders



os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'




class Agent:
    
    def __init__(self, model: str, openai_key: str, deepseek_key: str, langchain_key: str):
        os.environ['LANGCHAIN_API_KEY'] = langchain_key
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        os.environ['OPENAI_API_KEY'] = openai_key
                
        # self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=3000)
        self.text_splitter = None
        self.embeddings = OpenAIEmbeddings()
        self.mem_topk = 3
        self.doc_topk = 14
        self.def_topk = 5
        
        self.memory_buffer = ConversationBufferMemory(return_messages=True)
        self.response_counter = 0
        if 'deepseek_reasoner' in model:
            self.llm = ChatDeepSeek(model_name='deepseek-reasoner')
            print("dsr loaded")
        elif 'deepseek_chat' in model:
            self.llm = ChatDeepSeek(model_name='deepseek-chat')
            print("dsc loaded")
        elif 'openai' in model:
            self.llm = ChatOpenAI(model = 'o1-mini')
            print("oai loaded")
        else:
            print("Wrong input")
            
        self.loaders = DocLoaders()
        
        # self.response_counter = 0
        
    def set_text_split(self, size, overlap):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        
        
        
    def loader(self, file_path):
        docs = []
        if os.path.isfile(file_path): # MIGHT BE A POTENTIAL ISSUE BUT NOT USED SO FAR SO HAVE TO FIX LATER
            docs.extend(self.loaders.run(file_path))
            
        elif os.path.isdir(file_path):
            for file_name in os.listdir(file_path):
                path = os.path.join(file_path, file_name)
                docs.extend(self.loaders.run(path))
        else:
            print(f"'{file_path}' does not exist or is a special file type")
            
        return docs
    
        
    def build_prompt(self, context):
        memory_messages = self.memory_buffer.chat_memory.messages
        memory_text = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in memory_messages])

        template = ""
        if memory_text:
            template += "Chat Memory:\n{memory}\n\n"
        if len(context) != 0:
            template += "Context:\n{context}\n\n"
        template += "Answer the following question:\n{question}"
        
        # print(template)

        prompt = ChatPromptTemplate.from_template(template)
        return prompt.partial(memory=memory_text)
        
    def ask(self, question, context_path, mem_path):
        
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        
        # load the docs
        documents = self.loader(context_path)
        
        # load the mem
        # memory = self.loader(mem_path)
        
        # build prompt 
        prompt = self.build_prompt(documents)
        
        # create retrievers
        retriever = {}
        
        if len(documents) > 0:
            retriever["context"] = self.create_retriever("context", documents)
        
    
        
        retriever["question"] = RunnablePassthrough()
        rag_chain = (
            retriever
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        
        response = rag_chain.invoke(question)
        self.memory_buffer.chat_memory.add_user_message(question)
        self.memory_buffer.chat_memory.add_ai_message(response)
        return response
        
    def create_retriever(self, key, source):
        
        split_docs = self.text_splitter.split_documents(source)
        vector = Chroma.from_documents(split_docs, self.embeddings)
        
        if key == "context":
            return vector.as_retriever(search_kwargs={"k": self.doc_topk})
        elif key == "memory":
            return vector.as_retriever(search_kwargs={"k": self.mem_topk})
        else:
            return vector.as_retriever(search_kwargs={"k": self.def_topk})
        
    def reset(self, context_path, mem_path):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.memory_buffer.clear()
        self.clear_files(context_path, mem_path)
                
    def clear_files(self, context_path, mem_path):
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        for filename in os.listdir(context_path):
            file_path = os.path.join(context_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
        for filename in os.listdir(mem_path):
            file_path = os.path.join(mem_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        
        
        
