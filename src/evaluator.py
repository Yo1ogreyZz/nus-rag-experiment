"""
四类问题评测器
"""

import re

class Evaluator:
    """统一评测器"""
    
    @staticmethod
    def evaluate_factual(answer: str):
        """事实定位型评测"""
        
        answer_lower = answer.lower().strip()
        
        # 1. 长度评分（应该简洁）
        length = len(answer)
        if length <= 150:
            length_score = 1.0
        elif length <= 300:
            length_score = 0.8
        else:
            length_score = 0.5
        
        # 2. 包含具体信息（数字、URL、地址等）
        has_number = bool(re.search(r'\d+', answer))
        has_url = bool(re.search(r'http|www\.', answer_lower))
        has_specific_name = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', answer))
        
        specificity_score = (
            (0.4 if has_number else 0) +
            (0.3 if has_url else 0) +
            (0.3 if has_specific_name else 0)
        )
        
        # 3. 避免模糊回答
        vague_terms = ['maybe', 'might', 'possibly', 'unclear', "i don't know"]
        has_vague = any(term in answer_lower for term in vague_terms)
        clarity_score = 0.0 if has_vague else 1.0
        
        # 总分
        total_score = (
            0.3 * length_score +
            0.4 * specificity_score +
            0.3 * clarity_score
        )
        
        return {
            'category': 'factual',
            'length_score': round(length_score, 3),
            'specificity_score': round(specificity_score, 3),
            'clarity_score': round(clarity_score, 3),
            'total_score': round(total_score, 3),
            'answer_length': length,
            'has_specific_info': has_number or has_url or has_specific_name
        }
    
    @staticmethod
    def evaluate_procedural(answer: str):
        """过程解释型评测"""
        
        answer_lower = answer.lower()
        
        # 1. 步骤标记检测
        step_patterns = [
            r'\d+\.',  # 1. 2. 3.
            r'step \d+',  # Step 1, Step 2
            r'first|second|third|then|next|finally',  # 顺序词
        ]
        
        step_markers = 0
        for pattern in step_patterns:
            step_markers += len(re.findall(pattern, answer_lower))
        
        # 至少3个步骤算基本完整
        step_score = min(step_markers / 3.0, 1.0)
        
        # 2. 逻辑连接词
        connectors = ['first', 'then', 'next', 'after', 'finally', 'before']
        connector_count = sum(1 for c in connectors if c in answer_lower)
        logic_score = min(connector_count / 3.0, 1.0)
        
        # 3. 动作词（表示操作步骤）
        action_verbs = ['click', 'go', 'visit', 'submit', 'fill', 'select', 'open', 'enter']
        action_count = sum(1 for v in action_verbs if v in answer_lower)
        action_score = min(action_count / 4.0, 1.0)
        
        # 总分
        total_score = (
            0.4 * step_score +
            0.3 * logic_score +
            0.3 * action_score
        )
        
        return {
            'category': 'procedural',
            'step_score': round(step_score, 3),
            'logic_score': round(logic_score, 3),
            'action_score': round(action_score, 3),
            'total_score': round(total_score, 3),
            'step_markers_found': step_markers,
            'connectors_found': connector_count,
            'actions_found': action_count
        }
    
    @staticmethod
    def evaluate_comparative(answer: str):
        """比较分析型评测"""
        
        answer_lower = answer.lower()
        
        # 1. 对比词汇
        comparison_words = [
            'compare', 'difference', 'similar', 'both', 'while', 
            'whereas', 'however', 'in contrast', 'on the other hand'
        ]
        comparison_count = sum(1 for w in comparison_words if w in answer_lower)
        comparison_score = min(comparison_count / 3.0, 1.0)
        
        # 2. 多维度分析（至少提到2个对比维度）
        dimensions = ['cost', 'location', 'facility', 'time', 'quality', 'size', 'distance']
        dimension_count = sum(1 for d in dimensions if d in answer_lower)
        dimension_score = min(dimension_count / 2.0, 1.0)
        
        # 3. 结构化（是否分点讨论）
        has_structure = bool(re.search(r'\n\s*[-•*]|\d+\.', answer))
        structure_score = 1.0 if has_structure else 0.5
        
        # 4. 长度（比较型应该较详细）
        length = len(answer)
        if length >= 200:
            length_score = 1.0
        elif length >= 100:
            length_score = 0.7
        else:
            length_score = 0.4
        
        # 总分
        total_score = (
            0.3 * comparison_score +
            0.3 * dimension_score +
            0.2 * structure_score +
            0.2 * length_score
        )
        
        return {
            'category': 'comparative',
            'comparison_score': round(comparison_score, 3),
            'dimension_score': round(dimension_score, 3),
            'structure_score': round(structure_score, 3),
            'length_score': round(length_score, 3),
            'total_score': round(total_score, 3),
            'comparison_words_found': comparison_count,
            'dimensions_found': dimension_count
        }
    
    @staticmethod
    def evaluate_recommendation(answer: str):
        """约束推荐型评测"""
        
        answer_lower = answer.lower()
        
        # 1. 是否给出具体推荐
        has_recommendation = bool(re.search(r'recommend|suggest|should|could try', answer_lower))
        recommendation_score = 1.0 if has_recommendation else 0.0
        
        # 2. 是否提供理由
        reasoning_words = ['because', 'since', 'as', 'due to', 'offers', 'provides', 'has']
        reasoning_count = sum(1 for w in reasoning_words if w in answer_lower)
        reasoning_score = min(reasoning_count / 2.0, 1.0)
        
        # 3. 是否考虑约束条件
        constraint_words = ['budget', 'time', 'location', 'prefer', 'near', 'available', 'suitable']
        constraint_count = sum(1 for w in constraint_words if w in answer_lower)
        constraint_score = min(constraint_count / 2.0, 1.0)
        
        # 4. 是否提供多个选项
        option_markers = len(re.findall(r'\d+\)|option \d+|alternatively', answer_lower))
        diversity_score = min(option_markers / 2.0, 1.0)
        
        # 总分
        total_score = (
            0.3 * recommendation_score +
            0.3 * reasoning_score +
            0.25 * constraint_score +
            0.15 * diversity_score
        )
        
        return {
            'category': 'recommendation',
            'recommendation_score': round(recommendation_score, 3),
            'reasoning_score': round(reasoning_score, 3),
            'constraint_score': round(constraint_score, 3),
            'diversity_score': round(diversity_score, 3),
            'total_score': round(total_score, 3),
            'has_recommendation': has_recommendation,
            'reasoning_indicators': reasoning_count,
            'constraint_awareness': constraint_count
        }
    
    @staticmethod
    def evaluate(answer: str, question_type: str):
        """根据类型选择评测方法"""
        evaluators = {
            'factual': Evaluator.evaluate_factual,
            'procedural': Evaluator.evaluate_procedural,
            'comparative': Evaluator.evaluate_comparative,
            'recommendation': Evaluator.evaluate_recommendation
        }
        
        evaluator = evaluators.get(question_type)
        if not evaluator:
            raise ValueError(f"Unknown question type: {question_type}")
        
        return evaluator(answer)
