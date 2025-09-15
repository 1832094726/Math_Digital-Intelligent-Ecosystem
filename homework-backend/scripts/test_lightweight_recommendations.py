# -*- coding: utf-8 -*-
"""
测试轻量级深度学习推荐系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from services.enhanced_symbol_service import EnhancedSymbolService
from services.lightweight_dl_recommender import LightweightDeepLearningRecommender


def test_lightweight_recommendations():
    """测试轻量级深度学习推荐功能"""
    print("=" * 60)
    print("测试轻量级深度学习符号推荐系统")
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
    total_tests = len(test_cases)
    successful_tests = 0
    
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
                    model_type = rec.get('model_type', '')
                    print(f"    🤖 DL置信度: {dl_confidence:.3f}, 模型: {model_type}")
            
            print(f"📈 推荐来源统计: {sources}")
            print(f"🧠 深度学习推荐数量: {deep_learning_count}")
            
            # 检查是否包含期望的符号
            recommended_symbols = [rec.get('symbol', rec.get('latex', '')) for rec in recommendations]
            expected_found = sum(1 for exp in test_case['expected_symbols'] if exp in recommended_symbols)
            
            hit_rate = expected_found/len(test_case['expected_symbols'])*100
            print(f"✅ 期望符号命中率: {expected_found}/{len(test_case['expected_symbols'])} ({hit_rate:.1f}%)")
            
            if hit_rate >= 50:  # 命中率超过50%认为测试成功
                successful_tests += 1
                print("✅ 测试通过")
            else:
                print("⚠️ 测试部分通过")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print(f"\n📊 测试总结: {successful_tests}/{total_tests} 个测试通过")
    print("=" * 60)


def test_lightweight_model_directly():
    """直接测试轻量级深度学习模型"""
    print("\n" + "=" * 60)
    print("直接测试轻量级深度学习推荐模型")
    print("=" * 60)
    
    try:
        # 初始化轻量级深度学习推荐器
        dl_recommender = LightweightDeepLearningRecommender()
        print("✅ 轻量级深度学习推荐器初始化成功")
        
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
            {"id": 9, "symbol": "(", "latex": "(", "category": "括号"},
            {"id": 10, "symbol": ")", "latex": ")", "category": "括号"},
        ]
        
        test_contexts = [
            "解这个二次方程：x² + 3x + 2 = 0",
            "计算这个积分：∫x²dx",
            "求sin(x)的导数",
            "证明三角形ABC中的角度关系",
            "计算概率P(A∩B)"
        ]
        
        for i, context in enumerate(test_contexts, 1):
            print(f"\n🧪 测试 {i}: {context}")
            
            recommendations = dl_recommender.get_recommendations(
                user_id=f"test_user_{i}",
                context_text=context,
                candidate_symbols=candidate_symbols,
                top_k=5
            )
            
            print(f"📊 轻量级深度学习推荐结果:")
            for j, rec in enumerate(recommendations, 1):
                symbol = rec.get('symbol', '')
                score = rec.get('score', 0)
                model_type = rec.get('model_type', '')
                
                print(f"  {j}. {symbol} (分数: {score:.4f}, 模型: {model_type})")
        
        print("✅ 轻量级深度学习模型直接测试完成")
        
        # 测试模型保存和加载
        print("\n🔄 测试模型保存和加载...")
        dl_recommender.save_model()
        
        # 创建新实例并加载模型
        new_recommender = LightweightDeepLearningRecommender()
        new_recommender.load_model()
        
        print("✅ 模型保存和加载测试完成")
        
    except Exception as e:
        print(f"❌ 轻量级深度学习模型测试失败: {e}")


def analyze_context_features():
    """分析上下文特征提取"""
    print("\n" + "=" * 60)
    print("上下文特征分析")
    print("=" * 60)
    
    try:
        from services.lightweight_dl_recommender import ContextAnalyzer
        
        analyzer = ContextAnalyzer()
        
        test_texts = [
            "解这个二次方程：x² + 3x + 2 = 0",
            "计算这个积分：∫x²dx",
            "求sin(x)的导数",
            "证明三角形ABC中，∠A + ∠B + ∠C = 180°",
            "计算分数：3/4 + 1/2",
            "求极限：lim(x→0) sin(x)/x"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 文本 {i}: {text}")
            
            features = analyzer.analyze(text)
            
            print("🔍 提取的特征:")
            for feature, value in features.items():
                if isinstance(value, bool):
                    status = "✅" if value else "❌"
                    print(f"  {status} {feature}: {value}")
                else:
                    print(f"  📊 {feature}: {value:.3f}")
        
        print("✅ 上下文特征分析完成")
        
    except Exception as e:
        print(f"❌ 上下文特征分析失败: {e}")


if __name__ == "__main__":
    # 运行所有测试
    test_lightweight_recommendations()
    test_lightweight_model_directly()
    analyze_context_features()
    
    print("\n🎉 所有轻量级深度学习推荐测试完成!")
