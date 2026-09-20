import os
from dotenv import load_dotenv

# Document Loaders & Splitters
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Models & Vector Stores
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# Neo4j & Graph Transformers
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

try:
    # Primary import for updated packages
    from langchain_neo4j.graph_transformers import LLMGraphTransformer
except ImportError:
    # Fallback for experimental package
    from langchain_experimental.graph_transformers import LLMGraphTransformer
# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# ============================================================
# INGESTION FUNCTIONS
# ============================================================
def load_pdf(file_path: str):
    print("\n[1] Loading PDF Document...")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document page(s).")
    return documents

def split_documents(documents):
    print("\n[2A] Splitting documents into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    print("\n[3A] Embedding chunks and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="spotify_architecture"
    )
    print("Vector DB pipeline complete: ChromaDB populated in ./chroma_db.")
    return vectorstore

def extract_graph_information(documents):
    print("\n[2B] Extracting Entities and Relationships from Full Documents...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Service", "Database", "TechStack", "Feature", "Table", "Route"],
        allowed_relationships=["STORES_IN", "USES", "MANAGES", "DEPENDS_ON", "EXPOSES"],
        node_properties=["description"]
    )
    
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print(f"Graph Document extraction complete: Created {len(graph_documents)} objects.")
    return graph_documents

def store_in_neo4j(graph_documents):
    print("\n[3B] Storing Graph Documents in Neo4j...")
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    print("Knowledge Graph pipeline complete: Neo4j populated.")

# ============================================================
# HYBRID RAG ENGINE CLASS (Required by app.py)
# ============================================================
class HybridRAGEngine:
    def __init__(self):
        # 1. Connect to existing ChromaDB vector index
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings,
            collection_name="spotify_architecture"
        )
        
        # 2. Connect to Neo4j Graph Database
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        # 3. LLM Setup
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 4. Initialize Graph Cypher QA Chain
        self.cypher_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=False,
            allow_dangerous_requests=True
        )

    def query(self, question: str) -> dict:
        # A. Retrieve context from ChromaDB (Vector Search)
        vector_docs = self.vectorstore.similarity_search(question, k=3)
        vector_context = "\n\n".join([f"Source (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for doc in vector_docs])

        # B. Retrieve context from Neo4j (Knowledge Graph Query)
        try:
            graph_res = self.cypher_chain.invoke({"query": question})
            graph_context = str(graph_res.get("result", "No relational context found."))
        except Exception as e:
            graph_context = f"Graph retrieval failed or empty: {str(e)}"

        # C. Synthesize Final Answer using LLM
        prompt = f"""
You are an expert software architect answering questions about the Spotify Web App Architecture.

Use the following combined sources to answer the user's question accurately.

[Vector DB Semantic Search Context]:
{vector_context}

[Knowledge Graph Context]:
{graph_context}

Question: {question}

Answer:"""

        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "vector_context": vector_context,
            "graph_context": graph_context
        }

# ============================================================
# STANDALONE INGESTION EXECUTION
# ============================================================
if __name__ == "__main__":
    pdf_path = "../Data/spotify_web_app_architecture.pdf"
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")
        
    print("=" * 60)
    print("EXECUTING DUAL-BRANCH HYBRID RAG INGESTION PIPELINE")
    print("=" * 60)
    
    full_documents = load_pdf(pdf_path)
    
    # Branch 1: Vector Storage
    chunks = split_documents(full_documents)
    create_vector_store(chunks)
    
    # Branch 2: Knowledge Graph
    graph_documents = extract_graph_information(full_documents)
    store_in_neo4j(graph_documents)
    
    print("\n" + "=" * 60)
    print("INGESTION SUCCESSFULLY COMPLETED FOR BOTH BRANCHES")
    print("=" * 60)