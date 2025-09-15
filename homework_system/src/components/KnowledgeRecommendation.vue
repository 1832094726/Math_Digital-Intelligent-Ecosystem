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

    <div class="knowledge-content">
    
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
        <div style="display: flex; align-items: center; gap: 10px;">
          <small style="color: #909399;">
            题目: {{ currentProblem ? currentProblem.id : '无' }}
          </small>
          <el-switch
            v-model="showKnowledgeGraph"
            active-text="显示"
            inactive-text="隐藏"
            size="small"
          ></el-switch>
        </div>
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

    </div> <!-- 关闭 knowledge-content -->
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
      handler(newVal, oldVal) {
        console.log('KnowledgeRecommendation - currentProblem changed:', {
          newVal: newVal ? newVal.id : null,
          oldVal: oldVal ? oldVal.id : null,
          newContent: newVal ? newVal.content : null
        });

        if (newVal) {
          this.fetchRecommendations();

          // 如果知识图谱是显示状态，强制刷新
          if (this.showKnowledgeGraph) {
            console.log('KnowledgeRecommendation - currentProblem变化，强制刷新知识图谱');
            setTimeout(() => {
              this.forceRefreshGraph();
            }, 100);
          }
        } else {
          this.recommendedConcepts = [];
          this.learningPath = [];
          this.weakPoints = [];

          // 清空知识图谱
          if (this.graph) {
            this.graph.clear();
          }
        }
      },
      immediate: true
    },

    // 监听知识图谱显示状态
    showKnowledgeGraph: {
      handler(newVal) {
        console.log('KnowledgeRecommendation - showKnowledgeGraph changed:', newVal);
        if (newVal) {
          this.$nextTick(() => {
            console.log('KnowledgeRecommendation - 准备初始化知识图谱');
            this.initKnowledgeGraph();
          });
        }
      },
      immediate: true // 立即执行一次
    }
  },
  
  methods: {
    // 获取知识点推荐
    async fetchRecommendations() {
      console.log('KnowledgeRecommendation - fetchRecommendations 开始');
      console.log('KnowledgeRecommendation - currentProblem:', this.currentProblem);

      if (!this.currentProblem) {
        console.log('KnowledgeRecommendation - currentProblem 为空，退出');
        return;
      }

      this.loading = true;
      
      try {
        // 调用API获取知识点
        const knowledgeService = await import('../services/knowledgeService');
        
        // 如果题目有ID，通过ID获取知识点
        if (this.currentProblem.id) {
          const response = await knowledgeService.getQuestionKnowledgePoints(this.currentProblem.id);
          if (response.data && response.data.knowledge_points) {
            this.processKnowledgePoints(response.data.knowledge_points);
          } else {
            // 如果没有返回知识点，则通过题目内容获取
            const textResponse = await knowledgeService.getKnowledgePointsByText(this.currentProblem.content);
            if (textResponse.data && textResponse.data.knowledge_points) {
              this.processKnowledgePoints(textResponse.data.knowledge_points);
            } else {
              // 如果API没有返回知识点，使用模拟数据
              await this.simulateFetch();
            }
          }
        } else if (this.currentProblem.content) {
          // 如果题目没有ID但有内容，通过内容获取知识点
          const response = await knowledgeService.getKnowledgePointsByText(this.currentProblem.content);
          if (response.data && response.data.knowledge_points) {
            this.processKnowledgePoints(response.data.knowledge_points);
          } else {
            // 如果API没有返回知识点，使用模拟数据
            await this.simulateFetch();
          }
        } else {
          // 如果没有ID和内容，使用模拟数据
          await this.simulateFetch();
        }
        
        // 强制重新初始化知识图谱
        console.log('KnowledgeRecommendation - 检查是否需要初始化知识图谱');
        console.log('KnowledgeRecommendation - showKnowledgeGraph:', this.showKnowledgeGraph);
        if (this.showKnowledgeGraph) {
          console.log('KnowledgeRecommendation - 准备在nextTick中强制重新初始化知识图谱');
          // 使用setTimeout确保DOM更新完成
          setTimeout(() => {
            console.log('KnowledgeRecommendation - setTimeout中调用initKnowledgeGraph');
            this.initKnowledgeGraph();
          }, 50);
        }
      } catch (error) {
        console.error('KnowledgeRecommendation - fetchRecommendations失败:', error);
        this.$message.error('获取知识点推荐失败，使用本地数据');
        // 出错时使用模拟数据
        await this.simulateFetch();
      } finally {
        this.loading = false;
      }
    },
    
    // 处理API返回的知识点数据
    processKnowledgePoints(knowledgePoints) {
      if (!knowledgePoints || knowledgePoints.length === 0) {
        this.recommendedConcepts = [];
        return;
      }
      
      // 将后端知识点数据转换为前端所需格式
      this.recommendedConcepts = knowledgePoints.map(kp => ({
        id: kp.id,
        title: kp.name,
        description: kp.description || '暂无描述',
        examples: kp.examples ? kp.examples.map(ex => ex.problem + '\n' + ex.solution) : [],
        formulas: kp.key_points || [],
        relevance: 0.9
      }));
      
      // 生成学习路径
      this.generateLearningPath(knowledgePoints);
      
      // 检查用户弱点
      this.checkWeakPoints(knowledgePoints);
      
      // 默认展开第一个概念
      if (this.recommendedConcepts.length > 0) {
        this.activeConceptNames = [this.recommendedConcepts[0].title];
      }
    },
    
    // 生成学习路径
    generateLearningPath(knowledgePoints) {
      this.learningPath = [];
      
      // 根据知识点的先决条件生成学习路径
      for (const kp of knowledgePoints) {
        if (kp.prerequisites && kp.prerequisites.length > 0) {
          for (const prereq of kp.prerequisites) {
            this.learningPath.push({
              id: 'p' + Math.random().toString(36).substr(2, 9),
              title: prereq,
              description: `学习 ${prereq} 的基本概念和应用`
            });
          }
        }
        
        // 添加当前知识点
        this.learningPath.push({
          id: 'p' + kp.id,
          title: kp.name,
          description: kp.description || `掌握 ${kp.name} 的核心内容`
        });
        
        // 添加相关概念
        if (kp.related_concepts && kp.related_concepts.length > 0) {
          for (const related of kp.related_concepts) {
            this.learningPath.push({
              id: 'p' + Math.random().toString(36).substr(2, 9),
              title: related,
              description: `学习 ${related} 与 ${kp.name} 的关联`
            });
          }
        }
      }
      
      // 去重
      const uniquePaths = [];
      const seen = new Set();
      for (const path of this.learningPath) {
        if (!seen.has(path.title)) {
          seen.add(path.title);
          uniquePaths.push(path);
        }
      }
      this.learningPath = uniquePaths;
    },
    
    // 检查用户弱点
    checkWeakPoints(knowledgePoints) {
      this.weakPoints = [];
      
      // 如果用户上下文中有弱点数据
      if (this.userContext && this.userContext.weakPoints) {
        for (const kp of knowledgePoints) {
          const weakPoint = this.userContext.weakPoints.find(wp => 
            wp.name === kp.name || kp.related_concepts && kp.related_concepts.includes(wp.name)
          );
          
          if (weakPoint) {
            this.weakPoints.push({
              id: 'w' + kp.id,
              title: kp.name,
              description: `您在此知识点的掌握程度较弱，建议复习。`
            });
          }
        }
      }
      
      // 如果知识点有常见错误，也添加到弱点中
      for (const kp of knowledgePoints) {
        if (kp.common_errors && kp.common_errors.length > 0) {
          this.weakPoints.push({
            id: 'w' + kp.id,
            title: kp.name + ' - 常见错误',
            description: kp.common_errors.join('\n')
          });
        }
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
    async initKnowledgeGraph() {
      console.log('KnowledgeRecommendation - initKnowledgeGraph 开始');
      console.log('KnowledgeRecommendation - knowledgeGraphRef:', this.$refs.knowledgeGraphRef);
      console.log('KnowledgeRecommendation - recommendedConcepts:', this.recommendedConcepts);
      console.log('KnowledgeRecommendation - showKnowledgeGraph:', this.showKnowledgeGraph);

      if (!this.$refs.knowledgeGraphRef) {
        console.error('KnowledgeRecommendation - knowledgeGraphRef 不存在，延迟重试');
        // 延迟重试
        setTimeout(() => {
          if (this.$refs.knowledgeGraphRef && this.showKnowledgeGraph) {
            console.log('KnowledgeRecommendation - 延迟重试初始化知识图谱');
            this.initKnowledgeGraph();
          }
        }, 100);
        return;
      }

      try {
        console.log('KnowledgeRecommendation - 开始导入ECharts');
        // 动态导入ECharts
        const echarts = await import('echarts');
        console.log('KnowledgeRecommendation - ECharts导入成功');

        // 销毁之前的图表实例
        if (this.graph) {
          console.log('KnowledgeRecommendation - 销毁之前的图表实例');
          this.graph.dispose();
        }

        // 创建新的图表实例
        console.log('KnowledgeRecommendation - 创建新的图表实例');
        this.graph = echarts.init(this.$refs.knowledgeGraphRef);

        // 生成知识图谱数据
        console.log('KnowledgeRecommendation - 生成知识图谱数据');
        const graphData = this.generateKnowledgeGraphData();
        console.log('KnowledgeRecommendation - 图谱数据:', graphData);

        // 配置图表选项
        console.log('KnowledgeRecommendation - 配置图表选项');
        const option = {
          title: {
            text: '知识点关系图',
            left: 'center',
            textStyle: {
              fontSize: 14,
              color: '#333'
            }
          },
          tooltip: {
            trigger: 'item',
            formatter: function(params) {
              if (params.dataType === 'node') {
                return `<strong>${params.data.name}</strong><br/>${params.data.description || ''}`;
              } else {
                return `${params.data.source} → ${params.data.target}<br/>${params.data.relation}`;
              }
            }
          },
          series: [{
            type: 'graph',
            layout: 'force',
            symbolSize: 50,
            roam: true,
            label: {
              show: true,
              position: 'inside',
              fontSize: 10,
              color: '#fff'
            },
            edgeSymbol: ['circle', 'arrow'],
            edgeSymbolSize: [4, 10],
            data: graphData.nodes,
            links: graphData.links,
            categories: graphData.categories,
            force: {
              repulsion: 1000,
              gravity: 0.1,
              edgeLength: 100,
              layoutAnimation: true
            },
            lineStyle: {
              color: '#999',
              width: 2,
              curveness: 0.3
            },
            emphasis: {
              focus: 'adjacency',
              lineStyle: {
                width: 4
              }
            }
          }]
        };

        // 设置图表选项
        console.log('KnowledgeRecommendation - 设置图表选项');
        this.graph.setOption(option);
        console.log('KnowledgeRecommendation - 图表选项设置完成');

        // 监听窗口大小变化
        window.addEventListener('resize', () => {
          if (this.graph) {
            this.graph.resize();
          }
        });

        console.log('KnowledgeRecommendation - 知识图谱初始化完成');

      } catch (error) {
        console.error('KnowledgeRecommendation - 初始化知识图谱失败:', error);
        // 降级到简单显示
        const graphContainer = this.$refs.knowledgeGraphRef;
        graphContainer.innerHTML = '<div class="graph-placeholder"><p>知识图谱加载失败</p><p>错误: ' + error.message + '</p></div>';
      }
    },

    // 生成知识图谱数据
    generateKnowledgeGraphData() {
      const categories = [
        { name: '当前知识点', itemStyle: { color: '#409EFF' } },
        { name: '前置知识', itemStyle: { color: '#67C23A' } },
        { name: '相关知识', itemStyle: { color: '#E6A23C' } },
        { name: '后续知识', itemStyle: { color: '#F56C6C' } }
      ];

      // 基于当前题目生成知识图谱数据
      const nodes = [];
      const links = [];

      // 根据当前题目ID生成对应的知识点
      const knowledgePoints = this.getKnowledgePointsByQuestion();

      if (knowledgePoints.current.length > 0) {
        // 添加当前问题相关的知识点
        knowledgePoints.current.forEach((concept, index) => {
          nodes.push({
            id: concept.id,
            name: concept.name,
            description: concept.description,
            category: 0, // 当前知识点
            symbolSize: 60,
            x: Math.cos(index * 2 * Math.PI / knowledgePoints.current.length) * 150,
            y: Math.sin(index * 2 * Math.PI / knowledgePoints.current.length) * 150
          });
        });

        // 添加前置知识点
        knowledgePoints.prerequisites.forEach(pre => {
          nodes.push(pre);
          // 连接到第一个当前知识点
          if (knowledgePoints.current.length > 0) {
            links.push({
              source: pre.id,
              target: knowledgePoints.current[0].id,
              relation: '前置关系'
            });
          }
        });

        // 添加相关知识点
        knowledgePoints.related.forEach(rel => {
          nodes.push(rel);
          // 连接到相关的当前知识点
          if (knowledgePoints.current.length > 0) {
            links.push({
              source: knowledgePoints.current[0].id,
              target: rel.id,
              relation: '相关关系'
            });
          }
        });

        // 添加后续知识点
        knowledgePoints.advanced.forEach(adv => {
          nodes.push(adv);
          // 连接到当前知识点
          if (knowledgePoints.current.length > 0) {
            links.push({
              source: knowledgePoints.current[0].id,
              target: adv.id,
              relation: '后续关系'
            });
          }
        });

        // 添加知识点之间的关联
        if (knowledgePoints.current.length > 1) {
          for (let i = 0; i < knowledgePoints.current.length - 1; i++) {
            links.push({
              source: knowledgePoints.current[i].id,
              target: knowledgePoints.current[i + 1].id,
              relation: '关联关系'
            });
          }
        }
      } else {
        // 如果没有推荐知识点，显示默认图谱
        nodes.push(
          { id: 'default1', name: '选择题目', description: '请先选择一道题目', category: 0, symbolSize: 80 }
        );
      }

      return { nodes, links, categories };
    },

    // 根据当前题目获取对应的知识点
    getKnowledgePointsByQuestion() {
      if (!this.currentProblem) {
        return {
          current: [{ id: 'default1', name: '请选择题目', description: '选择一道题目查看相关知识点', category: 0, symbolSize: 80 }],
          prerequisites: [],
          related: [],
          advanced: []
        };
      }

      const questionId = this.currentProblem.id;
      const questionContent = this.currentProblem.content || '';

      // 根据题目ID和内容生成知识点
      switch (questionId) {
        case 'hw1_q1':
          return {
            current: [
              { id: 'curr1', name: '一元二次方程', description: '形如ax²+bx+c=0的方程', category: 0, symbolSize: 70 },
              { id: 'curr2', name: '求根公式', description: '使用求根公式解方程', category: 0, symbolSize: 60 }
            ],
            prerequisites: [
              { id: 'pre1', name: '一元一次方程', description: '线性方程的解法', category: 1, symbolSize: 50 },
              { id: 'pre2', name: '因式分解', description: '多项式分解技巧', category: 1, symbolSize: 50 }
            ],
            related: [
              { id: 'rel1', name: '二次函数', description: 'y=ax²+bx+c函数图像', category: 2, symbolSize: 50 },
              { id: 'rel2', name: '配方法', description: '完全平方式配方', category: 2, symbolSize: 50 }
            ],
            advanced: [
              { id: 'adv1', name: '高次方程', description: '三次及以上方程', category: 3, symbolSize: 50 },
              { id: 'adv2', name: '方程组', description: '多元方程组求解', category: 3, symbolSize: 50 }
            ]
          };

        case 'hw1_q2':
          return {
            current: [
              { id: 'curr3', name: '梯形面积', description: '梯形面积计算公式', category: 0, symbolSize: 70 },
              { id: 'curr4', name: '几何测量', description: '长度和面积的测量', category: 0, symbolSize: 60 }
            ],
            prerequisites: [
              { id: 'pre3', name: '矩形面积', description: '长方形面积公式', category: 1, symbolSize: 50 },
              { id: 'pre4', name: '基本运算', description: '加减乘除运算', category: 1, symbolSize: 50 }
            ],
            related: [
              { id: 'rel3', name: '平行四边形', description: '平行四边形面积', category: 2, symbolSize: 50 },
              { id: 'rel4', name: '三角形面积', description: '三角形面积公式', category: 2, symbolSize: 50 }
            ],
            advanced: [
              { id: 'adv3', name: '复合图形', description: '组合图形面积计算', category: 3, symbolSize: 50 },
              { id: 'adv4', name: '立体几何', description: '三维图形体积', category: 3, symbolSize: 50 }
            ]
          };

        case 'hw1_q3':
          return {
            current: [
              { id: 'curr5', name: '圆的面积', description: 'S=πr²圆面积公式', category: 0, symbolSize: 70 },
              { id: 'curr6', name: '圆周率π', description: '圆周率的概念和应用', category: 0, symbolSize: 60 }
            ],
            prerequisites: [
              { id: 'pre5', name: '圆的概念', description: '圆的基本性质', category: 1, symbolSize: 50 },
              { id: 'pre6', name: '平方运算', description: '数的平方计算', category: 1, symbolSize: 50 }
            ],
            related: [
              { id: 'rel5', name: '圆的周长', description: 'C=2πr周长公式', category: 2, symbolSize: 50 },
              { id: 'rel6', name: '扇形面积', description: '圆的部分面积', category: 2, symbolSize: 50 }
            ],
            advanced: [
              { id: 'adv5', name: '圆的方程', description: '解析几何中的圆', category: 3, symbolSize: 50 },
              { id: 'adv6', name: '球的体积', description: '三维圆形体积', category: 3, symbolSize: 50 }
            ]
          };

        case 'hw2_q1':
          return {
            current: [
              { id: 'curr7', name: '分数运算', description: '分数的加减乘除', category: 0, symbolSize: 70 },
              { id: 'curr8', name: '通分', description: '分数通分方法', category: 0, symbolSize: 60 }
            ],
            prerequisites: [
              { id: 'pre7', name: '分数概念', description: '分数的基本概念', category: 1, symbolSize: 50 },
              { id: 'pre8', name: '最小公倍数', description: '求最小公倍数', category: 1, symbolSize: 50 }
            ],
            related: [
              { id: 'rel7', name: '小数运算', description: '小数的四则运算', category: 2, symbolSize: 50 },
              { id: 'rel8', name: '百分数', description: '百分数与分数转换', category: 2, symbolSize: 50 }
            ],
            advanced: [
              { id: 'adv7', name: '分式方程', description: '含分数的方程', category: 3, symbolSize: 50 },
              { id: 'adv8', name: '比例应用', description: '分数在比例中应用', category: 3, symbolSize: 50 }
            ]
          };

        case 'hw2_q2':
          return {
            current: [
              { id: 'curr9', name: '百分比计算', description: '百分比的计算方法', category: 0, symbolSize: 70 },
              { id: 'curr10', name: '折扣问题', description: '商品折扣计算', category: 0, symbolSize: 60 }
            ],
            prerequisites: [
              { id: 'pre9', name: '百分数概念', description: '百分数的基本概念', category: 1, symbolSize: 50 },
              { id: 'pre10', name: '乘法运算', description: '小数乘法计算', category: 1, symbolSize: 50 }
            ],
            related: [
              { id: 'rel9', name: '利率计算', description: '银行利率问题', category: 2, symbolSize: 50 },
              { id: 'rel10', name: '增长率', description: '增长率计算', category: 2, symbolSize: 50 }
            ],
            advanced: [
              { id: 'adv9', name: '复利计算', description: '复合利率计算', category: 3, symbolSize: 50 },
              { id: 'adv10', name: '统计图表', description: '百分比在统计中应用', category: 3, symbolSize: 50 }
            ]
          };

        case 'hw2_q3':
          return {
            current: [
              { id: 'curr11', name: '三角形内角和', description: '三角形内角和为180°', category: 0, symbolSize: 70 },
              { id: 'curr12', name: '角度计算', description: '角度的加减运算', category: 0, symbolSize: 60 }
            ],
            prerequisites: [
              { id: 'pre11', name: '角的概念', description: '角的基本概念', category: 1, symbolSize: 50 },
              { id: 'pre12', name: '三角形概念', description: '三角形的基本性质', category: 1, symbolSize: 50 }
            ],
            related: [
              { id: 'rel11', name: '外角定理', description: '三角形外角性质', category: 2, symbolSize: 50 },
              { id: 'rel12', name: '特殊三角形', description: '等腰、等边三角形', category: 2, symbolSize: 50 }
            ],
            advanced: [
              { id: 'adv11', name: '三角函数', description: '正弦余弦正切', category: 3, symbolSize: 50 },
              { id: 'adv12', name: '解三角形', description: '利用三角函数解三角形', category: 3, symbolSize: 50 }
            ]
          };

        default:
          // 基于题目内容的通用知识点生成
          if (questionContent.includes('方程')) {
            return {
              current: [
                { id: 'curr_default1', name: '方程求解', description: '方程的基本解法', category: 0, symbolSize: 70 }
              ],
              prerequisites: [
                { id: 'pre_default1', name: '代数运算', description: '基本代数运算', category: 1, symbolSize: 50 }
              ],
              related: [
                { id: 'rel_default1', name: '函数关系', description: '方程与函数的关系', category: 2, symbolSize: 50 }
              ],
              advanced: [
                { id: 'adv_default1', name: '方程组', description: '多元方程组', category: 3, symbolSize: 50 }
              ]
            };
          } else if (questionContent.includes('面积') || questionContent.includes('周长')) {
            return {
              current: [
                { id: 'curr_default2', name: '几何测量', description: '图形的测量计算', category: 0, symbolSize: 70 }
              ],
              prerequisites: [
                { id: 'pre_default2', name: '基本图形', description: '基本几何图形', category: 1, symbolSize: 50 }
              ],
              related: [
                { id: 'rel_default2', name: '图形变换', description: '图形的变换', category: 2, symbolSize: 50 }
              ],
              advanced: [
                { id: 'adv_default2', name: '立体几何', description: '三维几何', category: 3, symbolSize: 50 }
              ]
            };
          } else {
            return {
              current: [
                { id: 'curr_default3', name: '数学问题', description: '当前数学问题', category: 0, symbolSize: 70 }
              ],
              prerequisites: [
                { id: 'pre_default3', name: '基础知识', description: '相关基础知识', category: 1, symbolSize: 50 }
              ],
              related: [
                { id: 'rel_default3', name: '相关概念', description: '相关数学概念', category: 2, symbolSize: 50 }
              ],
              advanced: [
                { id: 'adv_default3', name: '进阶内容', description: '进阶数学内容', category: 3, symbolSize: 50 }
              ]
            };
          }
      }
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
      if (this.graph) {
        this.graph.dispatchAction({
          type: 'graphRoam',
          zoom: 1.2
        });
      }
    },

    zoomOut() {
      if (this.graph) {
        this.graph.dispatchAction({
          type: 'graphRoam',
          zoom: 0.8
        });
      }
    },

    resetGraph() {
      if (this.graph) {
        this.graph.dispatchAction({
          type: 'restore'
        });
      }
    },

    // 强制刷新知识图谱
    forceRefreshGraph() {
      console.log('KnowledgeRecommendation - 强制刷新知识图谱');
      if (this.showKnowledgeGraph) {
        // 销毁现有图表
        if (this.graph) {
          this.graph.dispose();
          this.graph = null;
        }

        // 重新初始化
        this.$nextTick(() => {
          this.initKnowledgeGraph();
        });
      }
    }
  },

  // 组件销毁时清理图表实例
  beforeDestroy() {
    if (this.graph) {
      this.graph.dispose();
      this.graph = null;
    }
    // 移除窗口大小变化监听
    window.removeEventListener('resize', this.handleResize);
  },

  mounted() {
    // 绑定窗口大小变化处理函数
    this.handleResize = () => {
      if (this.graph) {
        this.graph.resize();
      }
    };

    // 确保在组件挂载后初始化知识图谱
    if (this.showKnowledgeGraph && this.currentProblem) {
      this.$nextTick(() => {
        console.log('KnowledgeRecommendation - mounted中初始化知识图谱');
        this.initKnowledgeGraph();
      });
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
  overflow: hidden;
}

.knowledge-content {
  flex: 1;
  overflow-y: auto;
  padding-right: 5px; /* 为滚动条留出空间 */
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
  height: 300px; /* 增加高度以更好显示图谱 */
  background-color: #fefefe;
  border: 1px solid #e6e6e6;
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.knowledge-graph {
  height: 100%;
  width: 100%;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
}

.graph-controls {
  position: absolute;
  bottom: 10px;
  right: 10px;
  z-index: 10;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 4px;
  padding: 4px;
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
