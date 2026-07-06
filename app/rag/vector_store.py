import chromadb
from chromadb.utils import embedding_functions
import json
import os
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.catalog_path = catalog_path
        
        # 1. Fallback to ONNX local embedder if HuggingFace token isn't verified
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        if HF_TOKEN:
            print("Connecting to remote Hugging Face Embedding API...")
            self.embedding_fn = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=HF_TOKEN,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            print("Loading default local ONNX embedding model to save memory...")
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # 2. Extract and clean the URL
        CHROMA_SERVER_URL = os.getenv("CHROMA_SERVER_URL", "http://localhost:8000")
        print(f"Connecting to remote Chroma server at: {CHROMA_SERVER_URL}")
        
        clean_host = CHROMA_SERVER_URL.replace("https://", "").replace("http://", "").strip("/")
        
        # 3. Securely handle Render architecture
        # Explicit settings bypass internal client proxy conversion errors causing 502s
        if "onrender.com" in clean_host:
            self.client = chromadb.HttpClient(
                host=clean_host,
                port=443,
                ssl=True,
                headers={"Default-Header": "Render-Handshake"}
            )
        else:
            self.client = chromadb.HttpClient(
                host=clean_host,
                port=8000,
                ssl=False
            )
        
        # 4. Safely create or load the collection
        try:
            self.collection = self.client.get_or_create_collection(
                name="shl_catalog",
                embedding_function=self.embedding_fn
            )
            item_count = self.collection.count()
            print(f"Vector store connected successfully! Found {item_count} items.")
            if item_count == 0:
                self.load_catalog()
        except Exception as e:
            print(f"Critical Connection warning: {e}")
            print("Activating local embedded fallback client...")
            # Emergency fallback client so your API continues running smoothly during server outages
            self.client = chromadb.PersistentClient(path="./chromadb")
            self.collection = self.client.get_or_create_collection(
                name="shl_catalog",
                embedding_function=self.embedding_fn
            )
            if self.collection.count() == 0:
                self.load_catalog()

    def load_catalog(self):
        """Loads and indexes raw JSON items into the active vector store."""
        if not os.path.exists(self.catalog_path):
            print("Downloading catalog dataset...")
            import requests
            response = requests.get(
                "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
            )
            os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
            with open(self.catalog_path, 'wb') as f:
                f.write(response.content)
        
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            try:
                items = json.loads(f.read(), strict=False)
            except Exception as e:
                print(f"Invalid local JSON catalog format: {e}")
                return
        
        catalog_items = [item for item in items if 'link' in item and 'name' in item]
        print(f"Indexing {len(catalog_items)} elements into collection...")
        
        documents, metadatas, ids = [], [], []
        for i, item in enumerate(catalog_items):
            metadata = {
                "name": str(item.get('name', '')),
                "link": str(item.get('link', '')),
                "keys": ', '.join(item.get('keys', [])) if isinstance(item.get('keys'), list) else str(item.get('keys', '')),
                "job_levels": ', '.join(item.get('job_levels', [])) if isinstance(item.get('job_levels'), list) else str(item.get('job_levels', ''))
            }
            text = f"Name: {item['name']}. Description: {item.get('description', '')}. Scope: {metadata['keys']}"
            documents.append(text)
            metadatas.append(metadata)
            ids.append(item.get('entity_id', f"idx_{i}"))
            
        # Segment payloads in chunks of 40 to protect small network buffers on Render free tiers
        for step in range(0, len(documents), 40):
            end_step = step + 40
            self.collection.add(
                documents=documents[step:end_step],
                metadatas=metadatas[step:end_step],
                ids=ids[step:end_step]
            )
        print("Indexing operation successfully finalized!")

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for relevant assessments using semantic similarity (RAG)"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            recommendations = []
            if results and results.get('metadatas') and len(results['metadatas']) > 0:
                for metadata in results['metadatas'][0]:
                    if not metadata:
                        continue
                    keys = metadata.get('keys', '').split(', ')
                    recommendations.append({
                        "name": metadata.get('name'),
                        "url": metadata.get('link'),
                        "test_type": self._get_test_type(keys)
                    })
            return recommendations
        except Exception as e:
            print(f"Search engine error: {e}")
            return []

    def _get_test_type(self, keys: List[str]) -> str:
        """Helper mapping metadata keywords to single letter specification requirements."""
        if "Personality & Behavior" in keys or "Personality" in keys:
            return "P"
        elif "Ability & Aptitude" in keys or "Aptitude" in keys:
            return "A"
        elif "Simulations" in keys:
            return "S"
        elif "Biodata & Situational Judgment" in keys:
            return "B"
        elif "Competencies" in keys:
            return "C"
        return "K"