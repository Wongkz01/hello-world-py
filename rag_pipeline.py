# c:/Users/R O G/vs_code_projects/rag_pipeline.py
import os
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate

def main():
    """
    Main function to set up and run the RAG pipeline.
    """
    # Load environment variables from .env file (for the OPENAI_API_KEY)
    load_dotenv()
    print("Loaded Hugging Face API Token.")

    # 1. Load the document
    loader = TextLoader("./data.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Split document into {len(texts)} chunks.")

    # 3. Create embeddings and store in a FAISS vector store
    # This uses a free, open-source model from Hugging Face that runs locally.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    print("Created FAISS vector store.")

    # 4. Set up the retrieval chain
    # This runs the LLM locally inside the container.
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
    )

    # 5. Create a custom prompt template
    # This template is simpler and easier for small models to understand.
    prompt_template = """
Use the following context to answer the question.

Context: {context}
Question: {question}

Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # This chain will retrieve documents and then use an LLM to answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" means it will "stuff" all context into the prompt
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt} # Use our custom prompt
    )
    print("RAG chain is ready.")
    print("Type 'exit' or 'quit' to stop.")

    # 6. Start an interactive loop
    while True:
        question = input("\nAsk a question: ")

        if question.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        try:
            result = qa_chain.invoke({"query": question})
            print("\n--- Answer ---")
            print(result["result"])
            print("----------------\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Exception Type: {type(e)}")
            print(f"Exception Args: {e.args}")
            print("Please check if your HUGGINGFACEHUB_API_TOKEN is valid.")

if __name__ == "__main__":
    main()
