<!--
  文件名: KnowledgeRecommendation.vue
  描述: 知识推荐模块组件，基于知识图谱和学生模型推荐相关知识点
  创建日期: 2023-06-15
-->

<template>
  <div class="knowledge-recommendation">
    <div class="module-header">
      <h3>知识点推荐</h3>
      <el-tooltip content="基于当前问题和您的学习状况，智能推荐相关知识点" placement="top">
        <i class="el-icon-question"></i>
      </el-tooltip>
    </div>
    
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="5" animated />
    </div>
    
    <div v-else-if="!currentProblem" class="empty-state">
      <i class="el-icon-document"></i>
      <p>请选择一道题目以获取相关知识点推荐</p>
    </div>
    
    <template v-else>
      <!-- 当前问题相关概念 -->
      <div class="section-header">
        <span>相关概念</span>
        <el-tag size="mini" type="primary">{{ recommendedConcepts.length }}</el-tag>
      </div>
      
      <div class="concepts-container">
        <el-collapse v-model="activeConceptNames" accordion>
          <el-collapse-item 
            v-for="concept in recommendedConcepts" 
            :key="concept.id"
            :title="concept.title"
            :name="concept.title"
          >
            <div class="concept-content">
              <div class="concept-description" v-html="concept.description"></div>
              
              <div class="concept-examples" v-if="concept.examples && concept.examples.length > 0">
                <div class="example-title">示例</div>
                <div 
                  v-for="(example, index) in concept.examples" 
                  :key="index"
                  class="example-item"
                  v-html="example"
                ></div>
              </div>
              
              <div class="concept-formulas" v-if="concept.formulas && concept.formulas.length > 0">
                <div class="formula-title">相关公式</div>
                <div 
                  v-for="(formula, index) in concept.formulas" 
                  :key="index"
                  class="formula-item"
                  v-html="formula"
                ></div>
              </div>
              
              <div class="concept-actions">
                <el-button 
                  type="primary" 
                  size="mini" 
                  @click="applyKnowledge(concept)"
                  icon="el-icon-check"
                >应用</el-button>
                <el-button 
                  type="info" 
                  size="mini" 
                  @click="viewDetails(concept)"
                  icon="el-icon-view"
                >详情</el-button>
              </div>
            </div>
          </el-collapse-item>
        </el-collapse>
      </div>
      
      <!-- 知识图谱可视化 -->
      <div class="section-header">
        <span>知识图谱</span>
        <el-switch 
          v-model="showKnowledgeGraph"
          active-text="显示"
          inactive-text="隐藏"
          size="small"
        ></el-switch>
      </div>
      
      <div v-if="showKnowledgeGraph" class="knowledge-graph-container">
        <div ref="knowledgeGraphRef" class="knowledge-graph"></div>
        <div class="graph-controls">
          <el-button-group size="mini">
            <el-button icon="el-icon-zoom-in" @click="zoomIn"></el-button>
            <el-button icon="el-icon-zoom-out" @click="zoomOut"></el-button>
            <el-button icon="el-icon-refresh" @click="resetGraph"></el-button>
          </el-button-group>
        </div>
      </div>
      
      <!-- 学习路径推荐 -->
      <div class="section-header">
        <span>学习路径</span>
        <el-tag size="mini" type="success">{{ learningPath.length }}</el-tag>
      </div>
      
      <div class="learning-path-container">
        <el-steps direction="vertical" :active="1">
          <el-step 
            v-for="(step, index) in learningPath" 
            :key="step.id"
            :title="step.title"
            :description="step.description"
            :status="getStepStatus(index)"
          ></el-step>
        </el-steps>
      </div>
      
      <!-- 历史弱点提示 -->
      <div v-if="weakPoints.length > 0" class="section-header">
        <span>需要注意的知识点</span>
        <el-tag size="mini" type="danger">{{ weakPoints.length }}</el-tag>
      </div>
      
      <div v-if="weakPoints.length > 0" class="weak-points-container">
        <el-alert
          v-for="point in weakPoints"
          :key="point.id"
          :title="point.title"
          :description="point.description"
          type="warning"
          show-icon
          :closable="false"
          style="margin-bottom: 10px;"
        >
          <el-button 
            slot="title" 
            size="mini" 
            type="text" 
            class="review-button"
            @click="reviewWeakPoint(point)"
          >
            立即复习
          </el-button>
        </el-alert>
      </div>
    </template>
  </div>
</template>

<script>
export default {
  name: 'KnowledgeRecommendation',
  
  props: {
    // 当前问题
    currentProblem: {
      type: Object,
      default: null
    },
    
    // 用户上下文
    userContext: {
      type: Object,
      required: true
    }
  },
  
  data() {
    return {
      // 加载状态
      loading: false,
      
      // 知识点推荐
      recommendedConcepts: [],
      
      // 展开的概念
      activeConceptNames: [],
      
      // 知识图谱显示
      showKnowledgeGraph: true,
      
      // 知识图谱实例
      graph: null,
      graphZoom: 1,
      
      // 学习路径
      learningPath: [],
      
      // 弱点知识点
      weakPoints: []
    };
  },
  
  watch: {
    // 监听当前问题变化，更新推荐
    currentProblem: {
      handler(newVal) {
        if (newVal) {
          this.fetchRecommendations();
        } else {
          this.recommendedConcepts = [];
          this.learningPath = [];
          this.weakPoints = [];
        }
      },
      immediate: true
    },
    
    // 监听知识图谱显示状态
    showKnowledgeGraph(newVal) {
      if (newVal) {
        this.$nextTick(() => {
          this.initKnowledgeGraph();
        });
      }
    }
  },
  
  methods: {
    // 获取知识点推荐
    async fetchRecommendations() {
      if (!this.currentProblem) return;
      
      this.loading = true;
      
      try {
        // 实际应用中应调用API获取推荐
        // 这里使用模拟数据
        await this.simulateFetch();
        
        // 初始化知识图谱
        if (this.showKnowledgeGraph) {
          this.$nextTick(() => {
            this.initKnowledgeGraph();
          });
        }
      } catch (error) {
        console.error('获取知识点推荐失败', error);
        this.$message.error('获取知识点推荐失败');
      } finally {
        this.loading = false;
      }
    },
    
    // 模拟获取数据
    async simulateFetch() {
      // 模拟API延迟
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // 根据当前问题生成模拟数据
      if (this.currentProblem.type === 'algebra') {
        this.recommendedConcepts = [
          {
            id: 'c1',
            title: '一元二次方程',
            description: '一元二次方程是指含有一个未知数，并且未知数的最高次数为2的方程。一般形式为：ax² + bx + c = 0，其中a、b、c是实数，且a ≠ 0。',
            examples: [
              '例如：3x² - 5x + 2 = 0',
              '例如：x² - 4 = 0'
            ],
            formulas: [
              '判别式：Δ = b² - 4ac',
              '求根公式：x = (-b ± √(b² - 4ac)) / (2a)'
            ],
            relevance: 0.95
          },
          {
            id: 'c2',
            title: '因式分解',
            description: '因式分解是将一个多项式表示成若干个多项式的乘积的形式。常用方法包括提取公因式、运用公式法、分组分解法等。',
            examples: [
              '例如：x² - 4 = (x - 2)(x + 2)',
              '例如：x² + 2x + 1 = (x + 1)²'
            ],
            formulas: [
              '平方差公式：a² - b² = (a - b)(a + b)',
              '完全平方公式：a² + 2ab + b² = (a + b)²',
              'a² - 2ab + b² = (a - b)²'
            ],
            relevance: 0.85
          },
          {
            id: 'c3',
            title: '配方法',
            description: '配方法是解一元二次方程的一种方法，通过恰当地变形，使方程中的二次项和一次项变成一个完全平方式。',
            examples: [
              '例如：将x² + 6x + 5 = 0变形为(x + 3)² = 4，然后求解'
            ],
            formulas: [
              '配方公式：x² + px = x² + px + (p/2)² - (p/2)² = (x + p/2)² - (p/2)²'
            ],
            relevance: 0.78
          }
        ];
        
        this.learningPath = [
          {
            id: 'p1',
            title: '一元二次方程的概念',
            description: '了解一元二次方程的定义和一般形式'
          },
          {
            id: 'p2',
            title: '因式分解法解方程',
            description: '掌握利用因式分解解一元二次方程'
          },
          {
            id: 'p3',
            title: '配方法解方程',
            description: '学习配方法的基本步骤和应用'
          },
          {
            id: 'p4',
            title: '公式法解方程',
            description: '掌握并灵活运用求根公式'
          }
        ];
        
        this.weakPoints = [
          {
            id: 'w1',
            title: '判别式的应用',
            description: '根据历史学习记录，您在使用判别式判断方程根的情况时存在困难'
          }
        ];
      } else if (this.currentProblem.type === 'geometry') {
        // 为几何问题生成不同的推荐
        this.recommendedConcepts = [
          {
            id: 'c4',
            title: '相似三角形',
            description: '相似三角形是指形状相同但大小可能不同的三角形。两个三角形相似，当且仅当它们的对应角相等，对应边成比例。',
            examples: [
              '例如：两个三角形，如果有两个角相等，则这两个三角形相似'
            ],
            formulas: [
              '相似三角形的对应边成比例：a/a\' = b/b\' = c/c\'',
              '相似三角形的面积比等于对应边长比的平方：S/S\' = (a/a\')²'
            ],
            relevance: 0.92
          },
          // 其他几何概念...
        ];
        
        // 其他几何相关数据...
      } else {
        // 默认推荐
        this.recommendedConcepts = [
          {
            id: 'c5',
            title: '基本数学概念',
            description: '针对当前问题的基本数学概念介绍',
            examples: ['示例问题1', '示例问题2'],
            formulas: ['相关公式1', '相关公式2'],
            relevance: 0.75
          }
        ];
      }
      
      // 根据用户上下文个性化推荐
      if (this.userContext.weakPoints && this.userContext.weakPoints.length > 0) {
        // 添加用户弱点相关的知识点
        this.weakPoints = this.userContext.weakPoints.map(wp => ({
          id: `w${wp.id}`,
          title: wp.name,
          description: `您在此知识点的掌握程度较弱，建议复习。掌握度: ${wp.proficiency}%`
        }));
      }
      
      // 随机决定是否展开第一个概念
      if (this.recommendedConcepts.length > 0 && Math.random() > 0.5) {
        this.activeConceptNames = [this.recommendedConcepts[0].title];
      }
    },
    
    // 初始化知识图谱可视化
    initKnowledgeGraph() {
      if (!this.$refs.knowledgeGraphRef) return;
      
      // 在实际应用中，应该使用可视化库如D3.js、ECharts或vis.js
      // 这里仅添加一个提示，表示该功能需要进一步实现
      const graphContainer = this.$refs.knowledgeGraphRef;
      graphContainer.innerHTML = '<div class="graph-placeholder"><p>知识图谱可视化</p><p>（此处将使用可视化库如D3.js绘制知识图谱）</p></div>';
    },
    
    // 应用知识点
    applyKnowledge(concept) {
      this.$emit('knowledge-selected', concept);
      this.$message.success(`已应用知识点：${concept.title}`);
    },
    
    // 查看知识点详情
    viewDetails(concept) {
      // 实际应用中应该跳转到知识点详情页或打开详情弹窗
      this.$alert(concept.description, concept.title, {
        dangerouslyUseHTMLString: true,
        confirmButtonText: '关闭'
      });
    },
    
    // 复习弱点知识点
    reviewWeakPoint(point) {
      // 实际应用中应该跳转到知识点学习页面
      this.$message.info(`即将开始复习：${point.title}`);
    },
    
    // 获取学习路径步骤状态
    getStepStatus(index) {
      if (index === 0) return 'success';
      if (index === 1) return 'process';
      return 'wait';
    },
    
    // 知识图谱缩放控制
    zoomIn() {
      this.graphZoom = Math.min(2, this.graphZoom + 0.1);
      this.updateGraphZoom();
    },
    
    zoomOut() {
      this.graphZoom = Math.max(0.5, this.graphZoom - 0.1);
      this.updateGraphZoom();
    },
    
    resetGraph() {
      this.graphZoom = 1;
      this.updateGraphZoom();
    },
    
    updateGraphZoom() {
      // 实际应用中应该调用图谱库的缩放方法
      console.log(`图谱缩放至 ${this.graphZoom}`);
    }
  }
};
</script>

<style scoped>
.knowledge-recommendation {
  height: 100%;
  padding: 15px;
  display: flex;
  flex-direction: column;
  overflow-y: hidden;
}

.module-header {
  display: flex;
  align-items: center;
  margin-bottom: 15px;
}

.module-header h3 {
  margin: 0;
  font-size: 18px;
  color: #303133;
  margin-right: 8px;
}

.module-header i {
  color: #909399;
  cursor: help;
}

.loading-state {
  padding: 10px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 150px;
  color: #909399;
}

.empty-state i {
  font-size: 36px;
  margin-bottom: 10px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 15px 0 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid #ebeef5;
  font-weight: 500;
  color: #303133;
}

.concepts-container {
  margin-bottom: 15px;
  overflow-y: auto;
  max-height: 300px;
}

.concept-content {
  padding: 5px;
}

.concept-description {
  margin-bottom: 10px;
  line-height: 1.5;
}

.example-title, .formula-title {
  font-weight: 500;
  margin-bottom: 5px;
  color: #303133;
}

.example-item, .formula-item {
  background-color: #f5f7fa;
  padding: 5px 10px;
  margin-bottom: 5px;
  border-radius: 4px;
  font-family: monospace;
}

.concept-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.knowledge-graph-container {
  position: relative;
  height: 200px;
  background-color: #f5f7fa;
  border-radius: 4px;
  margin-bottom: 15px;
  overflow: hidden;
}

.knowledge-graph {
  height: 100%;
  width: 100%;
}

.graph-controls {
  position: absolute;
  bottom: 10px;
  right: 10px;
}

.graph-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #909399;
  text-align: center;
}

.learning-path-container {
  margin-bottom: 15px;
  max-height: 200px;
  overflow-y: auto;
}

.weak-points-container {
  margin-bottom: 15px;
}

.review-button {
  margin-left: 10px;
}
</style>
