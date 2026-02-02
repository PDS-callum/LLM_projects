import argparse
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Literal, Tuple
from pydantic import BaseModel, Field

llm = ChatOllama(model="llama3.2")

class ChatBot(BaseModel):
    chat_history: List[Tuple[HumanMessage, AIMessage]] = Field(description="The chat history between the user and the chatbot.")
    question: str = Field(description="The question asked by the user.")
    answer: str = Field(description="The answer to the question.")
    finish_chat: Literal["Yes", "No"] = Field(description="Whether to finish the chat. ONLY EVER RETURN YES IF THE USER ASKS TO FINISH THE CHAT.")

chat_bot_llm = llm.with_structured_output(ChatBot)

def read_documents(files: str) -> List[Document]:
    """Load documents from file paths (comma-separated)."""
    documents = []
    file_list = [f.strip() for f in files.split(",")]
    
    for doc_path in file_list:
        if doc_path.endswith(".pdf"):
            loader = PyPDFLoader(doc_path)
            documents.extend(loader.load())
        else:
            raise ValueError(f"Unsupported file type: {doc_path}")
    
    return documents

def print_chat_bot_response(result: ChatBot):
    print(result.answer)

def main():
    parser = argparse.ArgumentParser(description="RAG chatbot that answers questions based on provided documents")
    parser.add_argument("--question", type=str, required=True, help="The question to ask")
    parser.add_argument("--files", type=str, required=True, help="Comma-separated list of document file paths")
    args = parser.parse_args()

    # Load and split documents
    documents = read_documents(args.files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = QdrantVectorStore.from_texts(
        texts=[doc.page_content for doc in splits],
        embedding=embeddings,
        metadatas=[doc.metadata for doc in splits],
        url="http://localhost:6333",
        collection_name="rag_documents",
    )
    retriever = vector_store.as_retriever()

    # Create prompt template
    template = """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template)

    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_bot_llm
    )

    messages = [HumanMessage(content=args.question)]
    finish_chat = "No"
    while finish_chat == "No":
        # Invoke the chain and print result
        result = rag_chain.invoke(f"Chat history: {messages}\nQuestion: {args.question}\nAnswer:")
        print_chat_bot_response(result)
        finish_chat = result.finish_chat
        messages.append(AIMessage(content=result.answer))
        args.question = input("Enter a question: ")
        messages.append(HumanMessage(content=args.question))

if __name__ == "__main__":
    main()
