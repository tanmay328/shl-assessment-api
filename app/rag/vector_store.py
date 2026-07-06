import chromadb
from chromadb.utils import embedding_functions
import json
import os
import re
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.catalog_path = catalog_path
        
        # 1. READ ENVIRONMENT VARIABLES FROM RENDER
        CHROMA_SERVER_URL = os.getenv("CHROMA_SERVER_URL", "http://localhost:8000")
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") 
        
        print("Connecting to remote Hugging Face Embedding API...")
        self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=HF_TOKEN,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print(f"Connecting to remote Chroma server at: {CHROMA_SERVER_URL}")
        
        # 2. DYNAMICALLY PARSE & CLEAN HOSTNAME TO PREVENT [Errno -5]
        # This strips out protocols and trailing paths, providing clean DNS resolution
        clean_host = CHROMA_SERVER_URL.replace("https://", "").replace("http://", "").strip("/")
        
        # If it contains 'onrender.com', we MUST connect using secure SSL on port 443
        is_ssl = "onrender.com" in clean_host
        
        # 3. INITIALIZE HTTP CLIENT EXPLICITLY
        self.client = chromadb.HttpClient(
            host=clean_host,
            ssl=is_ssl,
            port=443 if is_ssl else 8000
        )
        
        # Create or fetch collection on the standalone server
        self.collection = self.client.get_or_create_collection(
            name="shl_catalog",
            embedding_function=self.embedding_fn
        )
        
        # Check if the database needs initial population
        try:
            item_count = self.collection.count()
            if item_count == 0:
                self.load_catalog()
            else:
                print(f"Remote vector store already has {item_count} items")
        except Exception as e:
            print(f"Error checking or initializing collection status: {e}")
    
    def load_catalog(self):
        """Load catalog items into vector database"""
        if not os.path.exists(self.catalog_path):
            print("Downloading catalog...")
            import requests
            response = requests.get(
                "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
            )
            os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
            with open(self.catalog_path, 'wb') as f:
                f.write(response.content)
            print("Catalog downloaded successfully!")
        
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        try:
            items = json.loads(raw, strict=False)
        except json.JSONDecodeError as e:
            print(f"Catalog JSON invalid: {e}")
            raise
        
        catalog_items = [item for item in items if 'link' in item and 'name' in item]
        print(f"Indexing {len(catalog_items)} items into remote vector store...")
        
        documents = []
        metadatas = []
        ids = []
        for i, item in enumerate(catalog_items):
            metadata = {
                "name": str(item.get('name', '')),
                "link": str(item.get('link', '')),
                "description": str(item.get('description', ''))[:500],
                "keys": ', '.join(item.get('keys', [])) if isinstance(item.get('keys'), list) else str(item.get('keys', '')),
                "duration": str(item.get('duration', '')),
                "job_levels": ', '.join(item.get('job_levels', [])) if isinstance(item.get('job_levels'), list) else str(item.get('job_levels', ''))
            }
            text = f"{item['name']} - {item.get('description', '')} Keys: {metadata['keys']} Job Levels: {metadata['job_levels']}"
            documents.append(text)
            metadatas.append(metadata)
            ids.append(item.get('entity_id', f"id_{i}"))

        # Process in chunks of 50 to avoid network request limits
        batch_size = 50
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            self.collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
            print(f"  Indexed batch {start}-{min(end, len(documents))} of {len(documents)}")
        
        print(f"Successfully indexed {len(catalog_items)} items into remote vector store")
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for relevant assessments using semantic similarity (RAG)"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            recommendations = []
            if results['metadatas']:
                for metadata in results['metadatas'][0]:
                    keys = metadata.get('keys', '').split(', ')
                    test_type = self._get_test_type(keys)
                    recommendations.append({
                        "name": metadata.get('name'),
                        "url": metadata.get('link'),
                        "test_type": test_type
                    })
            
            return recommendations
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_assessment(self, name: str) -> Optional[Dict]:
        """Get assessment by name"""
        try:
            results = self.collection.get(where={"name": name})
            if results['metadatas'] and len(results['metadatas']) > 0:
                return results['metadatas'][0]
            return None
        except:
            return None
    
    def _get_test_type(self, keys: List[str]) -> str:
        """Get test type code from keys"""
        if "Personality & Behavior" in keys:
            return "P"
        elif "Ability & Aptitude" in keys:
            return "A"
        elif "Simulations" in keys:
            return "S"
        elif "Biodata & Situational Judgment" in keys:
            return "B"
        elif "Competencies" in keys:
            return "C"
        else:
            return "K"