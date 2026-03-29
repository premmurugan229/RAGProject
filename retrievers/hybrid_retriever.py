import weaviate
from langchain_community.vectorstores import Weaviate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from tavily import TavilyClient
from config import Config

class HybridRetriever:
    def __init__(self, config):
        self.config = config
        # Initialize Weaviate client
        self.weaviate_client = weaviate.connect_to_local()
        self.embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model=config.EMBEDDING_MODEL)
        # Note: In a real setup, you'd load actual documents here
        self.vector_store = Weaviate.from_texts(["sample docs"], embedding=self.embeddings, client=self.weaviate_client)
        
        self.neo4j_graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD
        )
        
        self.tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
    
    def retrieve(self, query: str):
        vector_docs = self.vector_store.similarity_search(query, k=3)
        # Use parameterized query to avoid injection
        kg_results = self.neo4j_graph.query(
            "MATCH (n)-[r]->(m) WHERE toLower(n.name) CONTAINS toLower($query) RETURN n,r,m LIMIT 5",
            {"query": query}
        )
        web_results = self.tavily.search(query=query, max_results=3)
        return {"vector": vector_docs, "kg": kg_results, "web": web_results}
