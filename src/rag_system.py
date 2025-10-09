"""
RAG系统核心
"""

import chromadb
import requests
import json

class OllamaClient:
    """Ollama客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def generate(self, model: str, prompt: str, temperature: float = 0.7):
        """生成回答"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'response': result.get('response', ''),
                    'model': model
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}",
                    'model': model
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model': model
            }

class RAGSystem:
    """RAG系统"""
    
    def __init__(self, db_path: str = "../data/vector_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection("nus_docs")
            print(f"✅ Loaded vector database from {db_path}")
        except:
            raise Exception(f"❌ Database not found! Please run build_vector_db.py first")
        
        self.ollama = OllamaClient()
    
    def retrieve(self, query: str, top_k: int = 5):
        """检索相关文档"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return retrieved_docs
    
    def build_prompt(self, question: str, retrieved_docs: list, question_type: str):
        """根据问题类型构建prompt"""
        
        context = "\n\n".join([
            f"[Document {i+1}]\n{doc['content']}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        type_instructions = {
            'factual': "Answer with a short, precise response. Include specific details like numbers, dates, or locations.",
            'procedural': "Provide a step-by-step guide. Number each step clearly.",
            'comparative': "Compare the options mentioned. Discuss similarities and differences across multiple dimensions.",
            'recommendation': "Give personalized recommendations based on the constraints mentioned. Explain your reasoning."
        }
        
        instruction = type_instructions.get(question_type, "Answer the question based on the context.")
        
        prompt = f"""You are a helpful assistant for NUS (National University of Singapore) students.

Context:
{context}

Question: {question}

Instructions: {instruction}

Answer:"""
        
        return prompt
    
    def query(self, question: str, model: str, question_type: str, top_k: int = 5):
        """完整RAG查询"""
        
        # 1. 检索
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # 2. 构建prompt
        prompt = self.build_prompt(question, retrieved_docs, question_type)
        
        # 3. 生成回答
        result = self.ollama.generate(model, prompt, temperature=0.7)
        
        return {
            'question': question,
            'question_type': question_type,
            'model': model,
            'retrieved_docs': retrieved_docs,
            'answer': result.get('response', ''),
            'success': result.get('success', False),
            'error': result.get('error', None)
        }
