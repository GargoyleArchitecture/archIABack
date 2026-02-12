
import os
import sys
from pathlib import Path
import sqlite3

# This script checks the sqlite3 file directly to see if any data exists
# since LangChain/Chroma wrappers might hide errors.

def check_sqlite(db_path):
    if not db_path.exists():
        print(f"[ERROR] {db_path} does not exist.")
        return
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"[INFO] Tables found: {tables}")
        
        if "collections" in tables:
            cursor.execute("SELECT name FROM collections;")
            cols = [c[0] for c in cursor.fetchall()]
            print(f"[INFO] Collections: {cols}")
            
        if "embeddings" in tables:
            cursor.execute("SELECT count(*) FROM embeddings;")
            count = cursor.fetchone()[0]
            print(f"[INFO] Total rows in 'embeddings' table: {count}")
            
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to read sqlite file: {e}")

def check_chroma_client():
    # Attempt to use chromadb directly to see collection count
    try:
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = str(Path(__file__).resolve().parent / "chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        
        print(f"[INFO] Connecting to Chroma at {persist_dir}")
        collections = client.list_collections()
        print(f"[INFO] Collection objects: {collections}")
        
        for col in collections:
            print(f"[INFO] Collection '{col.name}' has {col.count()} documents.")
            
    except ImportError:
        print("[WARN] chromadb not installed, skipping client check.")
    except Exception as e:
        print(f"[ERROR] Chroma client check failed: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    db_file = base_dir / "chroma_db" / "chroma.sqlite3"
    
    print("--- RAW SQLITE CHECK ---")
    check_sqlite(db_file)
    
    print("\n--- CHROMA CLIENT CHECK ---")
    check_chroma_client()
