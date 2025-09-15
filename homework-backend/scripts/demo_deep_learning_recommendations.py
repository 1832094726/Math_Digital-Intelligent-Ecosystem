# -*- coding: utf-8 -*-
"""
深度学习符号推荐系统演示
展示完整的推荐功能和集成效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.enhanced_symbol_service import EnhancedSymbolService
from services.lightweight_dl_recommender import LightweightDeepLearningRecommender


def demo_comprehensive_recommendations():
    """演示综合推荐功能"""
    print("🎯" + "=" * 60)
    print("深度学习符号推荐系统综合演示")
    print("🎯" + "=" * 60)
    
    # 初始化服务
    try:
        symbol_service = EnhancedSymbolService()
        print("✅ 符号推荐服务初始化成功")
    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")
        return
    
    # 演示场景
    demo_scenarios = [
        {
            "title": "🔢 代数方程求解",
            "user_id": 1,
            "question": "解二次方程 ax² + bx + c = 0",
            "input": "ax^2 + bx + c = 0",
            "description": "测试代数符号推荐"
        },
        {
            "title": "📐 几何图形证明", 
            "user_id": 2,
            "question": "证明：在直角三角形中，∠A + ∠B = 90°",
            "input": "∠A + ∠B = 90°",
            "description": "测试几何符号推荐"
        },
        {
            "title": "∫ 微积分计算",
            "user_id": 3,
            "question": "计算定积分 ∫₀¹ x² dx",
            "input": "\\int_0^1 x^2 dx",
            "description": "测试微积分符号推荐"
        },
        {
            "title": "📊 概率统计",
            "user_id": 4,
            "question": "计算条件概率 P(A|B) = P(A∩B)/P(B)",
            "input": "P(A|B) = P(A∩B)/P(B)",
            "description": "测试概率符号推荐"
        },
        {
            "title": "🔺 三角函数",
            "user_id": 5,
            "question": "求导数 d/dx[sin(x)cos(x)]",
            "input": "d/dx[sin(x)cos(x)]",
            "description": "测试三角函数符号推荐"
        }
    ]
    
    # 执行演示
    for i, scenario in enumerate(demo_scenarios, 1):
        print(f"\n{scenario['title']}")
        print("─" * 50)
        print(f"📝 题目: {scenario['question']}")
        print(f"⌨️  当前输入: {scenario['input']}")
        print(f"💡 说明: {scenario['description']}")
        
        try:
            # 获取推荐
            recommendations = symbol_service.get_recommendations(
                user_id=scenario['user_id'],
                question_text=scenario['question'],
                current_input=scenario['input'],
                limit=8
            )
            
            # 分析推荐结果
            sources = {}
            dl_count = 0
            high_score_count = 0
            
            print(f"\n🎯 推荐结果 (共{len(recommendations)}个):")
            
            for j, rec in enumerate(recommendations, 1):
                symbol = rec.get('symbol', '')
                source = rec.get('source', 'unknown')
                score = rec.get('score', 0)
                weighted_score = rec.get('weighted_score', score)
                
                sources[source] = sources.get(source, 0) + 1
                
                if source == 'deep_learning':
                    dl_count += 1
                    dl_confidence = rec.get('dl_confidence', 0)
                    model_type = rec.get('model_type', '')
                    print(f"  {j}. 🧠 {symbol} (分数: {score:.3f}, 加权: {weighted_score:.3f})")
                    print(f"     └─ 深度学习: {model_type}, 置信度: {dl_confidence:.3f}")
                else:
                    print(f"  {j}. {symbol} (分数: {score:.3f}, 加权: {weighted_score:.3f}, 来源: {source})")
                
                if weighted_score > 0.8:
                    high_score_count += 1
            
            # 统计信息
            print(f"\n📊 推荐统计:")
            print(f"  🧠 深度学习推荐: {dl_count}/{len(recommendations)} ({dl_count/len(recommendations)*100:.1f}%)")
            print(f"  ⭐ 高分推荐(>0.8): {high_score_count}/{len(recommendations)} ({high_score_count/len(recommendations)*100:.1f}%)")
            print(f"  📈 来源分布: {sources}")
            
            # 推荐质量评估
            if dl_count > 0:
                print(f"  ✅ 深度学习推荐已激活")
            else:
                print(f"  ⚠️  深度学习推荐未激活")
                
        except Exception as e:
            print(f"❌ 推荐失败: {e}")
    
    print("\n" + "🎯" + "=" * 60)
    print("演示完成")
    print("🎯" + "=" * 60)


def demo_deep_learning_features():
    """演示深度学习特有功能"""
    print("\n🤖" + "=" * 60)
    print("深度学习推荐器特性演示")
    print("🤖" + "=" * 60)
    
    try:
        # 直接使用深度学习推荐器
        dl_recommender = LightweightDeepLearningRecommender()
        print("✅ 轻量级深度学习推荐器初始化成功")
        
        # 测试上下文理解能力
        contexts = [
            "求解线性方程组",
            "计算矩阵的行列式", 
            "证明勾股定理",
            "求函数的极值",
            "计算概率分布"
        ]
        
        # 候选符号集
        symbols = [
            {"id": 1, "symbol": "=", "category": "等式"},
            {"id": 2, "symbol": "∫", "category": "积分"},
            {"id": 3, "symbol": "∠", "category": "角度"},
            {"id": 4, "symbol": "∩", "category": "集合"},
            {"id": 5, "symbol": "√", "category": "根式"},
            {"id": 6, "symbol": "∑", "category": "求和"},
            {"id": 7, "symbol": "π", "category": "常数"},
            {"id": 8, "symbol": "∞", "category": "无穷"}
        ]
        
        print("\n🧪 上下文适应性测试:")
        
        for i, context in enumerate(contexts, 1):
            print(f"\n{i}. 上下文: {context}")
            
            recommendations = dl_recommender.get_recommendations(
                user_id=f"demo_user_{i}",
                context_text=context,
                candidate_symbols=symbols,
                top_k=3
            )
            
            print("   推荐结果:")
            for j, rec in enumerate(recommendations, 1):
                symbol = rec.get('symbol', '')
                score = rec.get('score', 0)
                category = rec.get('category', '')
                print(f"     {j}. {symbol} ({category}) - 分数: {score:.4f}")
        
        # 测试用户个性化
        print("\n👤 用户个性化测试:")
        
        test_context = "解方程 x² + 2x + 1 = 0"
        
        for user_id in ["math_beginner", "math_expert", "geometry_lover"]:
            print(f"\n用户: {user_id}")
            
            recommendations = dl_recommender.get_recommendations(
                user_id=user_id,
                context_text=test_context,
                candidate_symbols=symbols,
                top_k=3
            )
            
            print("   个性化推荐:")
            for j, rec in enumerate(recommendations, 1):
                symbol = rec.get('symbol', '')
                score = rec.get('score', 0)
                print(f"     {j}. {symbol} - 分数: {score:.4f}")
        
        # 测试模型保存
        print("\n💾 模型持久化测试:")
        dl_recommender.save_model()
        print("✅ 模型保存成功")
        
        # 加载测试
        new_recommender = LightweightDeepLearningRecommender()
        new_recommender.load_model()
        print("✅ 模型加载成功")
        
        print("✅ 深度学习特性演示完成")
        
    except Exception as e:
        print(f"❌ 深度学习特性演示失败: {e}")


def demo_performance_analysis():
    """演示性能分析"""
    print("\n⚡" + "=" * 60)
    print("性能分析演示")
    print("⚡" + "=" * 60)
    
    try:
        symbol_service = EnhancedSymbolService()
        
        import time
        
        # 性能测试用例
        test_cases = [
            ("简单算术", "计算 2 + 3 = ?", "2 + 3 = "),
            ("代数方程", "解方程 x² - 4 = 0", "x^2 - 4 = 0"),
            ("微积分", "求导数 d/dx[x³]", "d/dx[x^3]"),
            ("复杂表达式", "计算 ∫₀^π sin(x)cos(x)dx", "\\int_0^\\pi sin(x)cos(x)dx")
        ]
        
        print("🔍 响应时间测试:")
        
        total_time = 0
        total_recommendations = 0
        
        for name, question, input_text in test_cases:
            start_time = time.time()
            
            recommendations = symbol_service.get_recommendations(
                user_id=1,
                question_text=question,
                current_input=input_text,
                limit=10
            )
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            total_time += response_time
            total_recommendations += len(recommendations)
            
            # 分析推荐来源
            dl_count = sum(1 for rec in recommendations if rec.get('source') == 'deep_learning')
            
            print(f"\n📊 {name}:")
            print(f"   ⏱️  响应时间: {response_time:.2f}ms")
            print(f"   📈 推荐数量: {len(recommendations)}")
            print(f"   🧠 深度学习: {dl_count}")
            print(f"   🎯 平均分数: {sum(rec.get('weighted_score', 0) for rec in recommendations)/len(recommendations):.3f}")
        
        print(f"\n📈 总体性能:")
        print(f"   ⏱️  平均响应时间: {total_time/len(test_cases):.2f}ms")
        print(f"   📊 平均推荐数量: {total_recommendations/len(test_cases):.1f}")
        print(f"   ✅ 性能表现: {'优秀' if total_time/len(test_cases) < 1000 else '良好' if total_time/len(test_cases) < 2000 else '需优化'}")
        
    except Exception as e:
        print(f"❌ 性能分析失败: {e}")


if __name__ == "__main__":
    # 运行完整演示
    demo_comprehensive_recommendations()
    demo_deep_learning_features()
    demo_performance_analysis()
    
    print("\n🎉 深度学习符号推荐系统演示完成!")
    print("🚀 系统已准备就绪，可以为用户提供智能符号推荐服务！")
