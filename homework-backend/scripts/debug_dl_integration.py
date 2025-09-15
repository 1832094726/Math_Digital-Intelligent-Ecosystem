# -*- coding: utf-8 -*-
"""
调试深度学习推荐集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.enhanced_symbol_service import EnhancedSymbolService

def debug_dl_integration():
    """调试深度学习推荐集成"""
    print("=" * 50)
    print("调试深度学习推荐集成")
    print("=" * 50)
    
    # 初始化服务
    service = EnhancedSymbolService()
    
    # 测试推荐
    result = service.get_recommendations(
        user_id=1, 
        question_text='解这个二次方程：x² + 3x + 2 = 0', 
        current_input='x^2 + 3x + 2 = 0'
    )
    
    print(f"推荐结果数量: {len(result)}")
    
    # 分析推荐来源
    sources = {}
    for rec in result:
        source = rec.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print(f"推荐来源统计: {sources}")
    
    # 显示前5个推荐
    print("\n前5个推荐:")
    for i, rec in enumerate(result[:5], 1):
        symbol = rec.get('symbol', '')
        source = rec.get('source', '')
        score = rec.get('score', 0)
        print(f"  {i}. {symbol} (来源: {source}, 分数: {score:.3f})")

if __name__ == "__main__":
    debug_dl_integration()
