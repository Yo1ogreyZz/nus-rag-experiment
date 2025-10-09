"""
构建向量数据库
用法: python build_vector_db.py
"""

import os
import chromadb
from pathlib import Path
import hashlib

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50):
    """文本分块"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_database(docs_dir: str = "../data/raw_docs", 
                   db_path: str = "../data/vector_db"):
    """构建向量数据库"""
    
    print(f"📚 Building vector database from: {docs_dir}")
    print(f"💾 Database will be saved to: {db_path}")
    
    # 初始化ChromaDB（使用默认embedding）
    client = chromadb.PersistentClient(path=db_path)
    
    # 删除旧collection（如果存在）
    try:
        client.delete_collection("nus_docs")
        print("🗑️  Deleted old collection")
    except:
        pass
    
    # 创建新collection
    collection = client.create_collection(
        name="nus_docs",
        metadata={"description": "NUS student life documents"}
    )
    
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"❌ Directory not found: {docs_dir}")
        print(f"📁 Creating directory...")
        docs_path.mkdir(parents=True, exist_ok=True)
        return
    
    # 支持的文件格式
    supported_extensions = ['.txt', '.md']
    all_files = []
    for ext in supported_extensions:
        all_files.extend(docs_path.glob(f'**/*{ext}'))
    
    if not all_files:
        print(f"⚠️  No documents found in {docs_dir}")
        print(f"📝 Please add .txt or .md files to {docs_dir}")
        return
    
    print(f"📄 Found {len(all_files)} documents")
    
    total_chunks = 0
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            # 分块
            chunks = chunk_text(content, chunk_size=300, overlap=50)
            
            # 为每个chunk生成唯一ID
            for idx, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(
                    f"{file_path.name}_{idx}".encode()
                ).hexdigest()
                
                collection.add(
                    documents=[chunk],
                    ids=[chunk_id],
                    metadatas=[{
                        'source': file_path.name,
                        'chunk_idx': idx
                    }]
                )
                total_chunks += 1
            
            print(f"  ✅ {file_path.name}: {len(chunks)} chunks")
            
        except Exception as e:
            print(f"  ❌ Error processing {file_path.name}: {e}")
    
    print(f"\n✨ Database built successfully!")
    print(f"📊 Total chunks indexed: {total_chunks}")
    print(f"💾 Saved to: {db_path}")

if __name__ == "__main__":
    build_database()
