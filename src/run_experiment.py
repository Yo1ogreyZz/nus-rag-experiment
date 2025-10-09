"""
主实验脚本
用法: python run_experiment.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rag_system import RAGSystem
from evaluator import Evaluator

def load_questions(questions_dir: str = "../questions"):
    """加载所有问题"""
    
    question_types = ['factual', 'procedural', 'comparative', 'recommendation']
    all_questions = {}
    
    questions_path = Path(questions_dir)
    if not questions_path.exists():
        print(f"📁 Creating questions directory: {questions_dir}")
        questions_path.mkdir(parents=True, exist_ok=True)
        
        # 创建示例问题文件
        examples = {
            'factual': [
                "What is the address of NUS Central Library?",
                "How many residential colleges does NUS have?",
                "What are the operating hours of MPSH gym?"
            ],
            'procedural': [
                "How do I apply for on-campus accommodation?",
                "What is the process for module registration?",
                "How do I book a study room in the library?"
            ],
            'comparative': [
                "What are the differences between Halls and Residential Colleges?",
                "Compare UTown Residence and PGPR.",
                "Compare Central Library and Science Library."
            ],
            'recommendation': [
                "Which hall should I choose if I want an active social life?",
                "I'm on a budget. What accommodation do you recommend?",
                "Where should I eat if I'm vegetarian?"
            ]
        }
        
        for qtype, questions in examples.items():
            filepath = questions_path / f"{qtype}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(questions))
            print(f"  ✅ Created example file: {qtype}.txt")
        
        print(f"\n📝 Please edit question files in {questions_dir}")
        print(f"   Each line = one question")
        return None
    
    # 加载问题
    for qtype in question_types:
        filepath = questions_path / f"{qtype}.txt"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
            all_questions[qtype] = questions
            print(f"✅ Loaded {len(questions)} {qtype} questions")
        else:
            print(f"⚠️  File not found: {qtype}.txt")
            all_questions[qtype] = []
    
    total = sum(len(q) for q in all_questions.values())
    print(f"📊 Total questions: {total}")
    
    return all_questions

def run_experiment(models: list, rag_system: RAGSystem, questions_dict: dict):
    """运行完整实验"""
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"../results/experiment_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Results will be saved to: {results_dir}")
    
    all_results = {}
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"🤖 Testing model: {model}")
        print(f"{'='*80}")
        
        model_results = {}
        
        for qtype, questions in questions_dict.items():
            if not questions:
                continue
            
            print(f"\n📋 Question type: {qtype.upper()}")
            
            type_results = []
            
            for question in tqdm(questions, desc=f"  Processing"):
                # RAG查询
                result = rag_system.query(question, model, qtype, top_k=5)
                
                if result['success']:
                    # 评测
                    evaluation = Evaluator.evaluate(result['answer'], qtype)
                    
                    # 合并结果
                    full_result = {
                        'question': question,
                        'answer': result['answer'],
                        'retrieved_docs_count': len(result['retrieved_docs']),
                        'evaluation': evaluation
                    }
                    
                    type_results.append(full_result)
                else:
                    print(f"\n    ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # 计算该类型平均分
            if type_results:
                avg_score = sum(r['evaluation']['total_score'] for r in type_results) / len(type_results)
                print(f"  📊 Average score: {avg_score:.3f}")
                
                model_results[qtype] = {
                    'questions_count': len(type_results),
                    'average_score': round(avg_score, 3),
                    'details': type_results
                }
        
        all_results[model] = model_results
        
        # 保存单个模型结果
        model_file = results_dir / f"{model.replace(':', '_')}.json"
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved: {model_file.name}")
    
    # 生成汇总报告
    generate_summary(all_results, results_dir)
    
    return all_results

def generate_summary(all_results: dict, results_dir: Path):
    """生成汇总报告"""
    
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")
    
    # 表格头
    print(f"{'Model':<20} {'Type':<20} {'Questions':<12} {'Avg Score':<12}")
    print("-"*80)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    for model, results in all_results.items():
        summary['models'][model] = {}
        
        for qtype, data in results.items():
            avg_score = data['average_score']
            count = data['questions_count']
            
            print(f"{model:<20} {qtype:<20} {count:<12} {avg_score:<12.3f}")
            
            summary['models'][model][qtype] = {
                'questions_count': count,
                'average_score': avg_score
            }
        
        print("-"*80)
    
    # 保存汇总
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print(f"✨ Experiment completed!")

def main():
    """主函数"""
    
    print("🚀 NUS RAG Experiment System")
    print("="*80)
    
    # 1. 加载问题
    questions_dict = load_questions()
    if questions_dict is None:
        return
    
    # 2. 初始化RAG系统
    try:
        rag_system = RAGSystem()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("📝 Please run: python build_vector_db.py first")
        return
    
    # 3. 定义要测试的模型
    models = [
        "qwen2.5:3b",
        "qwen2.5:1.5b",
        "llama3.2:3b",
        "phi3:mini",
        "deepseek-r1:1.5b"
    ]
    
    print(f"\n🤖 Models to test: {', '.join(models)}")
    
    # 4. 运行实验
    run_experiment(models, rag_system, questions_dict)

if __name__ == "__main__":
    main()
