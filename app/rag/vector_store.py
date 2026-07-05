import chromadb
from chromadb.utils import embedding_functions
import json
import os
import re
from typing import List, Dict, Optional

class VectorStore:
    def __init__(self, catalog_path: str = "data/catalog.json"):
        self.catalog_path = catalog_path
        
        # Initialize embedding function
        print("Loading embedding model...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chromadb")
        
        # Delete existing collection if it exists (to fix metadata issue)
        try:
            self.client.delete_collection("shl_catalog")
            print("Removed old collection with metadata issues")
        except:
            pass
        
        # Create new collection
        self.collection = self.client.get_or_create_collection(
            name="shl_catalog",
            embedding_function=self.embedding_fn
        )
        
        # Load catalog if collection is empty
        if self.collection.count() == 0:
            self.load_catalog()
        else:
            print(f"Vector store already has {self.collection.count()} items")
    
    def load_catalog(self):
        """Load catalog items into vector database"""
        # Download catalog if not exists
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
        
        # Load catalog — the file is freshly scraped from a live SHL endpoint
        # on every build/deploy, and has occasionally contained raw control
        # characters (e.g. literal newlines/tabs pasted directly into a
        # description field) inside string values. Strict JSON forbids this;
        # strict=False explicitly allows control characters inside strings,
        # which is exactly what we need here (blindly stripping \n/\t is
        # wrong since some of those are legitimate formatting inside text).
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            raw = f.read()
        try:
            items = json.loads(raw, strict=False)
        except json.JSONDecodeError as e:
            print(f"Catalog JSON invalid even with strict=False: {e}")
            raise
        
        # Filter for individual test solutions
        catalog_items = [item for item in items if 'link' in item and 'name' in item]
        
        print(f"Indexing {len(catalog_items)} items into vector store...")
        
        for i, item in enumerate(catalog_items):
            # Convert all metadata values to strings (ChromaDB requirement)
            metadata = {
                "name": str(item.get('name', '')),
                "link": str(item.get('link', '')),
                "description": str(item.get('description', ''))[:500],
                "keys": ', '.join(item.get('keys', [])),
                "duration": str(item.get('duration', '')),
                "job_levels": ', '.join(item.get('job_levels', []))
            }
            
            # Create rich text for embedding
            text = f"{item['name']} - {item.get('description', '')} " \
                   f"Keys: {', '.join(item.get('keys', []))} " \
                   f"Job Levels: {', '.join(item.get('job_levels', []))}"
            
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[item.get('entity_id', f"id_{i}")]
            )
        
        print(f"Successfully indexed {len(catalog_items)} items into vector store")
    
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