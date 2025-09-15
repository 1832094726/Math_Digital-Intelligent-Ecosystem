# -*- coding: utf-8 -*-
"""
测试深度学习推荐系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from services.enhanced_symbol_service import EnhancedSymbolService
from services.deep_learning_recommender import DeepLearningRecommender


def test_deep_learning_recommendations():
    """测试深度学习推荐功能"""
    print("=" * 60)
    print("测试深度学习符号推荐系统")
    print("=" * 60)
    
    # 初始化服务
    try:
        symbol_service = EnhancedSymbolService()
        print("✅ 符号服务初始化成功")
    except Exception as e:
        print(f"❌ 符号服务初始化失败: {e}")
        return
    
    # 测试用例
    test_cases = [
        {
            "name": "二次方程求解",
            "user_id": 1,
            "question_text": "解这个二次方程：x² + 3x + 2 = 0",
            "current_input": "x^2 + 3x + 2 = 0",
            "expected_symbols": ["=", "²", "+", "x"]
        },
        {
            "name": "积分计算",
            "user_id": 2,
            "question_text": "计算这个积分：∫x²dx",
            "current_input": "\\int x^2 dx",
            "expected_symbols": ["∫", "²", "dx", "x"]
        },
        {
            "name": "三角函数",
            "user_id": 3,
            "question_text": "求sin(x)的导数",
            "current_input": "sin(x)",
            "expected_symbols": ["sin", "(", ")", "x", "'"]
        },
        {
            "name": "几何证明",
            "user_id": 4,
            "question_text": "证明三角形ABC中，∠A + ∠B + ∠C = 180°",
            "current_input": "∠A + ∠B + ∠C = 180°",
            "expected_symbols": ["∠", "+", "=", "°"]
        },
        {
            "name": "概率计算",
            "user_id": 5,
            "question_text": "计算P(A∩B) = P(A) × P(B|A)",
            "current_input": "P(A∩B)",
            "expected_symbols": ["P", "(", ")", "∩", "×", "|"]
        }
    ]
    
    # 执行测试
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {test_case['name']}")
        print(f"题目: {test_case['question_text']}")
        print(f"当前输入: {test_case['current_input']}")
        
        try:
            # 获取推荐结果
            recommendations = symbol_service.get_recommendations(
                user_id=test_case['user_id'],
                question_text=test_case['question_text'],
                current_input=test_case['current_input'],
                limit=10
            )
            
            print(f"📊 获得 {len(recommendations)} 个推荐结果")
            
            # 分析推荐来源
            sources = {}
            deep_learning_count = 0
            
            for rec in recommendations[:5]:  # 只显示前5个
                source = rec.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                
                if source == 'deep_learning':
                    deep_learning_count += 1
                
                symbol = rec.get('symbol', rec.get('latex', ''))
                score = rec.get('score', 0)
                weighted_score = rec.get('weighted_score', score)
                
                print(f"  🔸 {symbol} (分数: {score:.3f}, 加权: {weighted_score:.3f}, 来源: {source})")
                
                # 显示深度学习特有信息
                if source == 'deep_learning':
                    dl_confidence = rec.get('dl_confidence', 0)
                    model_explanation = rec.get('model_explanation', '')
                    print(f"    🤖 DL置信度: {dl_confidence:.3f}, 解释: {model_explanation}")
            
            print(f"📈 推荐来源统计: {sources}")
            print(f"🧠 深度学习推荐数量: {deep_learning_count}")
            
            # 检查是否包含期望的符号
            recommended_symbols = [rec.get('symbol', rec.get('latex', '')) for rec in recommendations]
            expected_found = sum(1 for exp in test_case['expected_symbols'] if exp in recommended_symbols)
            
            print(f"✅ 期望符号命中率: {expected_found}/{len(test_case['expected_symbols'])} ({expected_found/len(test_case['expected_symbols'])*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("深度学习推荐系统测试完成")
    print("=" * 60)


def test_deep_learning_model_directly():
    """直接测试深度学习模型"""
    print("\n" + "=" * 60)
    print("直接测试深度学习推荐模型")
    print("=" * 60)
    
    try:
        # 初始化深度学习推荐器
        dl_recommender = DeepLearningRecommender()
        print("✅ 深度学习推荐器初始化成功")
        
        # 准备测试数据
        candidate_symbols = [
            {"id": 1, "symbol": "=", "latex": "=", "category": "运算符"},
            {"id": 2, "symbol": "²", "latex": "^2", "category": "指数"},
            {"id": 3, "symbol": "+", "latex": "+", "category": "运算符"},
            {"id": 4, "symbol": "x", "latex": "x", "category": "变量"},
            {"id": 5, "symbol": "∫", "latex": "\\int", "category": "积分"},
            {"id": 6, "symbol": "sin", "latex": "\\sin", "category": "三角函数"},
            {"id": 7, "symbol": "∠", "latex": "\\angle", "category": "几何"},
            {"id": 8, "symbol": "°", "latex": "^\\circ", "category": "单位"},
        ]
        
        test_contexts = [
            "解这个二次方程：x² + 3x + 2 = 0",
            "计算这个积分：∫x²dx",
            "求sin(x)的导数",
            "证明三角形ABC中的角度关系"
        ]
        
        for i, context in enumerate(test_contexts, 1):
            print(f"\n🧪 测试 {i}: {context}")
            
            recommendations = dl_recommender.get_recommendations(
                user_id=f"test_user_{i}",
                context_text=context,
                candidate_symbols=candidate_symbols,
                top_k=5
            )
            
            print(f"📊 深度学习推荐结果:")
            for j, rec in enumerate(recommendations, 1):
                symbol = rec.get('symbol', '')
                score = rec.get('score', 0)
                model_type = rec.get('model_type', '')
                
                print(f"  {j}. {symbol} (分数: {score:.4f}, 模型: {model_type})")
        
        print("✅ 深度学习模型直接测试完成")
        
    except Exception as e:
        print(f"❌ 深度学习模型测试失败: {e}")


def analyze_recommendation_performance():
    """分析推荐性能"""
    print("\n" + "=" * 60)
    print("推荐性能分析")
    print("=" * 60)
    
    try:
        symbol_service = EnhancedSymbolService()
        
        # 性能测试用例
        performance_tests = [
            {
                "name": "简单表达式",
                "question": "计算 2 + 3 = ?",
                "input": "2 + 3 = ",
                "complexity": "低"
            },
            {
                "name": "复杂方程",
                "question": "解方程 ax² + bx + c = 0",
                "input": "ax^2 + bx + c = 0",
                "complexity": "中"
            },
            {
                "name": "高等数学",
                "question": "计算 ∫₀^∞ e^(-x²) dx",
                "input": "\\int_0^\\infty e^{-x^2} dx",
                "complexity": "高"
            }
        ]
        
        for test in performance_tests:
            print(f"\n📊 {test['name']} (复杂度: {test['complexity']})")
            
            import time
            start_time = time.time()
            
            recommendations = symbol_service.get_recommendations(
                user_id=1,
                question_text=test['question'],
                current_input=test['input'],
                limit=15
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            # 分析推荐质量
            sources = {}
            avg_score = 0
            
            for rec in recommendations:
                source = rec.get('source', 'unknown')
                sources[source] = sources.get(source, 0) + 1
                avg_score += rec.get('weighted_score', 0)
            
            if recommendations:
                avg_score /= len(recommendations)
            
            print(f"  ⏱️  响应时间: {response_time:.2f}ms")
            print(f"  📈 推荐数量: {len(recommendations)}")
            print(f"  🎯 平均分数: {avg_score:.3f}")
            print(f"  📊 来源分布: {sources}")
        
        print("✅ 性能分析完成")
        
    except Exception as e:
        print(f"❌ 性能分析失败: {e}")


if __name__ == "__main__":
    # 运行所有测试
    test_deep_learning_recommendations()
    test_deep_learning_model_directly()
    analyze_recommendation_performance()
    
    print("\n🎉 所有测试完成!")
