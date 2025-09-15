// 交互功能扩展
class DatabaseInteractions {
    constructor(visualization) {
        this.viz = visualization;
        this.init();
    }

    init() {
        this.setupAdvancedInteractions();
    }

    setupAdvancedInteractions() {
        // 扩展可视化类的方法
        this.viz.zoomIn = () => {
            const scale = this.viz.network.getScale();
            this.viz.network.moveTo({ scale: scale * 1.2 });
        };

        this.viz.zoomOut = () => {
            const scale = this.viz.network.getScale();
            this.viz.network.moveTo({ scale: scale * 0.8 });
        };

        this.viz.resetView = () => {
            this.viz.network.fit();
        };

        // 添加双击事件监听
        this.viz.network.on('doubleClick', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                this.showTableDetailModal(nodeId);
            }
        });

        // 添加点击空白区域事件
        this.viz.network.on('click', (params) => {
            if (params.nodes.length === 0 && params.edges.length === 0) {
                this.hideTableDetailModal();
            }
        });

        this.viz.toggleModule = (module, visible) => {
            if (visible) {
                this.viz.visibleModules.add(module);
            } else {
                this.viz.visibleModules.delete(module);
            }
            this.updateVisibleTables();
        };

        this.viz.changeLayout = (layoutType) => {
            let layoutOptions = {};
            switch (layoutType) {
                case 'hierarchical':
                    layoutOptions = {
                        hierarchical: {
                            enabled: true,
                            direction: 'UD',
                            sortMethod: 'directed',
                            nodeSpacing: 200,
                            levelSeparation: 150
                        }
                    };
                    break;
                case 'force':
                    layoutOptions = {
                        hierarchical: { enabled: false }
                    };
                    this.viz.network.setOptions({ physics: { enabled: true } });
                    break;
                case 'circular':
                    this.arrangeCircular();
                    return;
                case 'grid':
                    this.arrangeGrid();
                    return;
            }
            this.viz.network.setOptions({ layout: layoutOptions });
        };

        this.viz.performSearch = () => {
            const query = document.getElementById('tableSearch').value.toLowerCase();
            this.searchTables(query);
        };

        this.viz.closeModal = () => {
            document.getElementById('tableModal').style.display = 'none';
        };

        this.viz.exportDiagram = () => {
            this.exportToImage();
        };

        this.viz.toggleFullscreen = () => {
            this.toggleFullscreen();
        };

        this.viz.highlightConnectedNodes = (nodeId) => {
            this.highlightConnectedNodes(nodeId);
        };
    }

    updateVisibleTables() {
        const visibleTables = [];
        const visibleEdges = [];

        // 过滤可见的表
        Object.values(databaseSchema.tables).forEach(table => {
            if (this.viz.visibleModules.has(table.category)) {
                visibleTables.push(table.name);
            }
        });

        // 过滤可见的关系
        databaseSchema.relationships.forEach((rel, index) => {
            if (visibleTables.includes(rel.from) && visibleTables.includes(rel.to)) {
                visibleEdges.push(`edge_${index}`);
            }
        });

        // 更新网络显示
        const allNodes = this.viz.nodes.get();
        const allEdges = this.viz.edges.get();

        allNodes.forEach(node => {
            this.viz.nodes.update({
                id: node.id,
                hidden: !visibleTables.includes(node.id)
            });
        });

        allEdges.forEach(edge => {
            this.viz.edges.update({
                id: edge.id,
                hidden: !visibleEdges.includes(edge.id)
            });
        });

        // 更新侧边栏显示
        document.querySelectorAll('.category').forEach(category => {
            const categoryType = category.dataset.category;
            if (categoryType) {
                category.style.display = this.viz.visibleModules.has(categoryType) ? 'block' : 'none';
            }
        });
    }

    arrangeCircular() {
        const positions = {};
        const tables = Object.keys(databaseSchema.tables);
        const radius = 300;
        const centerX = 0;
        const centerY = 0;

        tables.forEach((tableName, index) => {
            const angle = (index / tables.length) * 2 * Math.PI;
            positions[tableName] = {
                x: centerX + Math.cos(angle) * radius,
                y: centerY + Math.sin(angle) * radius
            };
        });

        this.viz.network.setOptions({ physics: { enabled: false } });
        this.viz.network.setPositions(positions);
    }

    arrangeGrid() {
        const positions = {};
        const tables = Object.keys(databaseSchema.tables);
        const cols = Math.ceil(Math.sqrt(tables.length));
        const spacing = 250;

        tables.forEach((tableName, index) => {
            const row = Math.floor(index / cols);
            const col = index % cols;
            positions[tableName] = {
                x: col * spacing - (cols * spacing) / 2,
                y: row * spacing - (Math.ceil(tables.length / cols) * spacing) / 2
            };
        });

        this.viz.network.setOptions({ physics: { enabled: false } });
        this.viz.network.setPositions(positions);
    }

    searchTables(query) {
        const results = [];
        const resultsContainer = document.getElementById('searchResults');

        if (!query.trim()) {
            resultsContainer.innerHTML = '';
            return;
        }

        // 搜索表名和字段
        Object.values(databaseSchema.tables).forEach(table => {
            let matches = [];

            // 搜索表名
            if (table.name.toLowerCase().includes(query) || 
                table.displayName.toLowerCase().includes(query)) {
                matches.push({ type: 'table', text: table.displayName });
            }

            // 搜索字段
            table.fields.forEach(field => {
                if (field.name.toLowerCase().includes(query) || 
                    field.description.toLowerCase().includes(query)) {
                    matches.push({ type: 'field', text: `${field.name} - ${field.description}` });
                }
            });

            if (matches.length > 0) {
                results.push({
                    table: table,
                    matches: matches
                });
            }
        });

        // 显示搜索结果
        let html = '';
        if (results.length === 0) {
            html = '<div class="search-no-results">未找到匹配结果</div>';
        } else {
            results.forEach(result => {
                html += `
                    <div class="search-result-item" data-table="${result.table.name}">
                        <div class="search-result-table">${result.table.displayName}</div>
                        <div class="search-result-matches">
                            ${result.matches.map(match => 
                                `<span class="search-match ${match.type}">${match.text}</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
            });
        }

        resultsContainer.innerHTML = html;

        // 添加点击事件
        resultsContainer.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const tableName = e.currentTarget.dataset.table;
                this.viz.selectTable(tableName);
                this.focusOnTable(tableName);
            });
        });
    }

    focusOnTable(tableName) {
        const nodePosition = this.viz.network.getPositions([tableName]);
        if (nodePosition[tableName]) {
            this.viz.network.moveTo({
                position: nodePosition[tableName],
                scale: 1.5,
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }
    }

    highlightConnectedNodes(nodeId) {
        // 临时高亮连接的节点
        const connectedEdges = this.viz.edges.get().filter(edge => 
            edge.from === nodeId || edge.to === nodeId
        );

        const connectedNodes = new Set();
        connectedEdges.forEach(edge => {
            if (edge.from !== nodeId) connectedNodes.add(edge.from);
            if (edge.to !== nodeId) connectedNodes.add(edge.to);
        });

        // 添加临时高亮样式
        connectedNodes.forEach(connectedNodeId => {
            const table = databaseSchema.tables[connectedNodeId];
            const module = databaseSchema.modules[table.category];
            this.viz.nodes.update({
                id: connectedNodeId,
                color: {
                    background: this.viz.lightenColor(module.color, 40),
                    border: module.color
                }
            });
        });

        // 3秒后恢复
        setTimeout(() => {
            if (this.viz.selectedTable !== nodeId) {
                this.viz.resetHighlight();
            }
        }, 3000);
    }

    exportToImage() {
        // 创建导出选项对话框
        const exportOptions = `
            <div class="export-dialog">
                <h4>导出图表</h4>
                <div class="export-options">
                    <label>
                        <input type="radio" name="format" value="png" checked> PNG图片
                    </label>
                    <label>
                        <input type="radio" name="format" value="svg"> SVG矢量图
                    </label>
                    <label>
                        <input type="radio" name="format" value="pdf"> PDF文档
                    </label>
                </div>
                <div class="export-settings">
                    <label>
                        图片质量:
                        <select id="exportQuality">
                            <option value="1">标准</option>
                            <option value="2" selected>高清</option>
                            <option value="3">超高清</option>
                        </select>
                    </label>
                    <label>
                        <input type="checkbox" id="includeBackground" checked> 包含背景
                    </label>
                </div>
                <div class="export-actions">
                    <button onclick="this.performExport()" class="btn btn-primary">导出</button>
                    <button onclick="this.closeExportDialog()" class="btn btn-secondary">取消</button>
                </div>
            </div>
        `;

        // 显示导出对话框
        const dialog = document.createElement('div');
        dialog.className = 'export-modal';
        dialog.innerHTML = exportOptions;
        document.body.appendChild(dialog);

        // 绑定导出功能
        dialog.querySelector('.btn-primary').onclick = () => {
            this.performExport(dialog);
        };
        dialog.querySelector('.btn-secondary').onclick = () => {
            document.body.removeChild(dialog);
        };
    }

    performExport(dialog) {
        const format = dialog.querySelector('input[name="format"]:checked').value;
        const quality = parseInt(dialog.querySelector('#exportQuality').value);
        const includeBackground = dialog.querySelector('#includeBackground').checked;

        const canvas = this.viz.network.canvas.frame.canvas;
        const ctx = canvas.getContext('2d');

        if (format === 'png') {
            // 导出PNG
            const link = document.createElement('a');
            link.download = `database-schema-${new Date().getTime()}.png`;
            link.href = canvas.toDataURL('image/png', quality);
            link.click();
        } else if (format === 'svg') {
            // 导出SVG (需要额外实现)
            this.exportSVG();
        } else if (format === 'pdf') {
            // 导出PDF (需要额外实现)
            this.exportPDF();
        }

        document.body.removeChild(dialog);
    }

    exportSVG() {
        // SVG导出实现
        console.log('SVG导出功能待实现');
    }

    exportPDF() {
        // PDF导出实现
        console.log('PDF导出功能待实现');
    }

    toggleFullscreen() {
        const container = document.querySelector('.visualization-area');

        if (!document.fullscreenElement) {
            container.requestFullscreen().then(() => {
                container.classList.add('fullscreen');
                document.getElementById('fullscreenBtn').innerHTML = '<i class="fas fa-compress"></i> 退出全屏';
            });
        } else {
            document.exitFullscreen().then(() => {
                container.classList.remove('fullscreen');
                document.getElementById('fullscreenBtn').innerHTML = '<i class="fas fa-expand"></i> 全屏';
            });
        }
    }

    // 表详情面板功能
    showTableDetailModal(tableName) {
        const table = databaseSchema.tables[tableName];
        if (!table) return;

        // 显示右侧详情面板
        this.showTableDetailPanel(table);
    }

    hideTableDetailModal() {
        // 隐藏右侧详情面板
        this.hideTableDetailPanel();
    }

    showTableDetailPanel(table) {
        console.log(`📋 显示表 ${table.name} 的详情面板`);

        // 获取或创建详情面板
        let detailPanel = document.querySelector('.table-detail-panel');
        if (!detailPanel) {
            detailPanel = this.createTableDetailPanel();
            document.querySelector('.main-content').appendChild(detailPanel);
        }

        // 更新面板内容
        this.updateTableDetailPanel(detailPanel, table);

        // 显示面板
        detailPanel.classList.add('show');

        // 调整主内容区域布局
        document.querySelector('.visualization-area').classList.add('with-detail-panel');
    }

    hideTableDetailPanel() {
        const detailPanel = document.querySelector('.table-detail-panel');
        if (detailPanel) {
            detailPanel.classList.remove('show');
            document.querySelector('.visualization-area').classList.remove('with-detail-panel');
        }
    }

    createTableDetailPanel() {
        const panel = document.createElement('div');
        panel.className = 'table-detail-panel';
        panel.innerHTML = `
            <div class="panel-header">
                <h3 class="panel-title">表详情</h3>
                <button class="panel-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="panel-content">
                <div class="loading-spinner">
                    <i class="fas fa-spinner fa-spin"></i> 加载中...
                </div>
            </div>
        `;

        // 添加关闭事件
        const closeBtn = panel.querySelector('.panel-close');
        closeBtn.addEventListener('click', () => {
            this.hideTableDetailPanel();
        });

        return panel;
    }

    updateTableDetailPanel(panel, table) {
        const module = databaseSchema.modules[table.category];

        panel.querySelector('.panel-title').textContent = table.displayName;

        const content = panel.querySelector('.panel-content');
        content.innerHTML = `
            <div class="table-overview">
                <div class="table-header-info">
                    <h4>${table.displayName}</h4>
                    <span class="module-badge" style="background: ${module.color}">${module.name}</span>
                </div>
                <p class="table-description">${table.description}</p>
                <div class="table-stats-mini">
                    <span class="stat-mini">字段: ${table.fields.length}</span>
                    <span class="stat-mini">关系: ${table.relationships.length}</span>
                    <span class="stat-mini">键: ${table.fields.filter(f => f.isPrimary || f.isForeignKey).length}</span>
                </div>
            </div>

            <div class="panel-tabs">
                <button class="panel-tab-btn active" data-tab="fields">字段结构</button>
                <button class="panel-tab-btn" data-tab="relationships">关系说明</button>
                <button class="panel-tab-btn" data-tab="data">实时数据</button>
            </div>

            <div class="panel-tab-content">
                <div class="panel-tab-pane active" id="panel-fields-tab">
                    ${this.renderFieldsList(table.fields)}
                </div>

                <div class="panel-tab-pane" id="panel-relationships-tab">
                    ${this.renderRelationshipsExplanation(table)}
                </div>

                <div class="panel-tab-pane" id="panel-data-tab">
                    <div class="data-controls-mini">
                        <button class="btn-mini btn-primary" onclick="window.interactions.refreshPanelData('${table.name}')">
                            <i class="fas fa-sync-alt"></i> 刷新
                        </button>
                        <span class="data-count-mini" id="panel-data-count">加载中...</span>
                    </div>
                    <div class="panel-data-container">
                        <div class="loading-spinner">
                            <i class="fas fa-spinner fa-spin"></i> 加载数据中...
                        </div>
                    </div>
                </div>
            </div>
        `;

        // 添加标签页切换事件
        panel.querySelectorAll('.panel-tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.target.dataset.tab;
                this.switchPanelTab(panel, tabName);
            });
        });

        // 加载实时数据
        this.loadPanelTableData(table, panel);
    }





    renderFieldsList(fields) {
        return `
            <div class="fields-list">
                ${fields.map(field => `
                    <div class="field-item">
                        <div class="field-header">
                            <span class="field-name">
                                ${field.name}
                                ${field.isPrimary ? '<span class="badge-mini primary">PK</span>' : ''}
                                ${field.isForeignKey ? '<span class="badge-mini foreign">FK</span>' : ''}
                            </span>
                            <span class="field-type" style="color: ${window.databaseAPI.getFieldTypeColor(field.type)}">
                                ${field.type}
                            </span>
                        </div>
                        <div class="field-description">${field.description || '无描述'}</div>
                        ${field.defaultValue ? `<div class="field-default">默认值: ${field.defaultValue}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    renderRelationshipsExplanation(table) {
        if (table.relationships.length === 0) {
            return '<div class="empty-explanation">该表没有定义关系</div>';
        }

        return `
            <div class="relationships-explanation">
                <h5>关系说明</h5>
                ${table.relationships.map(rel => `
                    <div class="relationship-item">
                        <div class="relationship-header">
                            <span class="relationship-type-badge ${rel.type}">${this.getRelationshipTypeText(rel.type)}</span>
                            <span class="relationship-target">${rel.table}</span>
                        </div>
                        <div class="relationship-description">${rel.description}</div>
                        <div class="relationship-explanation-text">
                            ${this.getRelationshipExplanation(table.name, rel)}
                        </div>
                        ${rel.foreignKey ? `<div class="relationship-key">外键: ${rel.foreignKey}</div>` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    getRelationshipTypeText(type) {
        const typeMap = {
            'belongsTo': '属于',
            'hasMany': '拥有多个',
            'hasOne': '拥有一个',
            'one-to-one': '一对一',
            'one-to-many': '一对多',
            'many-to-many': '多对多'
        };
        return typeMap[type] || type;
    }

    getRelationshipExplanation(tableName, relationship) {
        const explanations = {
            'users': {
                'user_sessions': '🔐 为什么需要会话管理？\n• 支持多设备同时登录（手机、电脑、平板）\n• 实现安全的登录状态跟踪和超时控制\n• 防止账号被盗用，提供登录历史审计\n• 支持"踢出其他设备"等安全功能',
                'notifications': '📢 为什么需要通知系统？\n• 及时通知学生作业发布、截止时间提醒\n• 老师可以推送重要公告和学习资料\n• 个性化推送，避免信息过载\n• 支持不同类型通知的优先级管理',
                'homework_submissions': '📝 为什么分离用户和提交？\n• 同一个作业可以被多个学生提交，需要独立记录\n• 支持作业的多次提交和版本管理\n• 便于统计分析学生的提交习惯和成绩趋势\n• 实现作业的批量评阅和成绩管理',
                'homework_progress': '⏱️ 为什么需要进度跟踪？\n• 支持作业的断点续做，提升用户体验\n• 记录学生的答题时间和思考过程\n• 为智能推荐提供数据支持\n• 帮助老师了解学生的学习难点',
                'class_students': '👥 为什么用户和班级分离？\n• 学生可能转班、升级，需要灵活的关系管理\n• 支持一个学生同时属于多个班级（如兴趣班）\n• 便于班级管理和学生信息的批量操作\n• 实现精细化的权限控制和数据隔离'
            },
            'homeworks': {
                'questions': '❓ 为什么作业和题目分离？\n• 同一道题目可以在多个作业中复用，避免重复录入\n• 支持题目的版本管理和难度调整\n• 便于构建题库系统和智能组卷\n• 实现题目的标签化管理和快速检索',
                'homework_assignments': '📋 为什么需要分配表？\n• 同一个作业模板可以分配给不同班级，设置不同截止时间\n• 支持个性化的作业分配（如分层教学）\n• 便于统计不同班级的完成情况\n• 实现作业的批量分配和管理',
                'homework_submissions': '✅ 为什么需要提交记录？\n• 记录每个学生的提交时间、答案和成绩\n• 支持作业的自动批改和人工复核\n• 为学习分析提供原始数据\n• 实现成绩的统计分析和排名',
                'homework_progress': '📊 为什么需要进度管理？\n• 实时跟踪学生的答题进度和停留时间\n• 为自适应学习提供数据支持\n• 帮助识别学生的薄弱环节\n• 支持学习路径的个性化推荐'
            },
            'schools': {
                'grades': '🏫 为什么学校和年级分离？\n• 不同学校的年级设置可能不同（如九年一贯制）\n• 支持年级的动态调整和学制改革\n• 便于按年级进行教学管理和资源分配\n• 实现跨年级的教学活动和评比',
                'classes': '🎓 为什么需要班级管理？\n• 实现学生的分组管理和差异化教学\n• 支持班级间的竞争和协作活动\n• 便于教学资源的精准投放\n• 为家校沟通提供组织基础',
                'curriculum_standards': '📚 为什么支持多课程标准？\n• 不同地区可能采用不同的教学大纲\n• 支持国际化教学和双语教育\n• 便于教学内容的标准化管理\n• 实现课程的个性化定制和优化'
            },
            'classes': {
                'class_students': '👨‍🎓 为什么班级和学生分离？\n• 支持学生的转班、休学、复学等状态变化\n• 便于班级人数的动态管理和优化\n• 实现学生信息的批量导入和更新\n• 为班级活动和评比提供数据基础',
                'class_schedules': '📅 为什么需要课程表？\n• 合理安排教学时间和教室资源\n• 避免课程冲突和资源浪费\n• 支持个性化的课程安排\n• 为智能排课提供数据支持',
                'homework_assignments': '📝 为什么按班级分配作业？\n• 实现差异化教学和分层作业\n• 便于作业的批量管理和统计\n• 支持班级间的教学进度协调\n• 为教学效果评估提供数据'
            },
            'courses': {
                'course_modules': '📚 为什么课程需要模块化？\n• 一门数学课程包含多个模块：代数、几何、统计等\n• 便于按模块进行教学进度管理和考核\n• 支持学生按模块选择学习内容\n• 实现课程的个性化定制和难度分层',
                'lessons': '📖 为什么课程包含多个课时？\n• 一门课程需要分解为多个课时来完成教学\n• 每个课时有具体的教学目标和内容\n• 便于教学进度的精细化管理\n• 支持学生的分步学习和复习'
            },
            'course_modules': {
                'chapters': '📑 为什么模块包含多个章节？\n• 一个代数模块包含：方程、不等式、函数等章节\n• 每个章节有独立的知识点和学习目标\n• 便于知识的结构化组织和渐进学习\n• 支持章节级别的测试和评估',
                'lessons': '🎯 为什么模块对应多个课时？\n• 一个几何模块需要多个课时来完成教学\n• 不同课时可以采用不同的教学方法\n• 便于教学资源的合理分配\n• 支持教学效果的跟踪和调整'
            },
            'chapters': {
                'lessons': '📝 为什么章节包含多个课时？\n• 一个"二次函数"章节需要多个课时：概念、图像、应用等\n• 每个课时专注于特定的知识点或技能\n• 便于学生的分步掌握和巩固\n• 支持差异化教学和个性化辅导'
            },
            'grades': {
                'classes': '🎓 为什么年级包含多个班级？\n• 七年级可能有：七(1)班、七(2)班、七(3)班等\n• 便于学生的分组管理和差异化教学\n• 支持班级间的良性竞争和协作\n• 实现教学资源的合理配置',
                'subjects': '📚 为什么年级对应多个学科？\n• 七年级学生需要学习：数学、语文、英语、物理等\n• 不同学科有不同的教学要求和进度\n• 便于跨学科的教学协调和管理\n• 支持学生的全面发展和综合评价'
            },
            'subjects': {
                'courses': '📖 为什么学科包含多个课程？\n• 数学学科包含：代数、几何、概率统计等课程\n• 每个课程有独立的教学大纲和目标\n• 便于专业化教学和师资配置\n• 支持学生的兴趣选择和能力发展'
            }
        };

        const tableExplanations = explanations[tableName];
        if (tableExplanations && tableExplanations[relationship.table]) {
            return tableExplanations[relationship.table];
        }

        // 通用解释 - 基于关系类型提供具体的业务场景说明
        return this.getGenericRelationshipExplanation(tableName, relationship);
    }

    getGenericRelationshipExplanation(tableName, relationship) {
        const targetTable = relationship.table;
        const relType = relationship.type;

        // 基于关系类型和表名生成具体的业务场景说明
        if (relType === 'belongsTo') {
            return `🔗 为什么是多对一关系？\n• 举例：多个${this.getTableDisplayName(tableName)}记录属于同一个${this.getTableDisplayName(targetTable)}\n• 比如：多个学生属于同一个班级，多个课时属于同一门课程\n• 这样设计是为了避免数据冗余，实现数据的规范化存储\n• 便于统一管理和批量操作，如按${this.getTableDisplayName(targetTable)}统计${this.getTableDisplayName(tableName)}信息`;
        } else if (relType === 'hasMany') {
            return `📊 为什么是一对多关系？\n• 举例：一个${this.getTableDisplayName(tableName)}可以包含多个${this.getTableDisplayName(targetTable)}\n• 比如：一个班级包含多个学生，一门课程包含多个课时\n• 这样设计支持层次化的数据组织和管理\n• 便于实现分组统计、批量操作和权限控制`;
        } else if (relType === 'hasOne') {
            return `🎯 为什么是一对一关系？\n• 举例：一个${this.getTableDisplayName(tableName)}对应一个${this.getTableDisplayName(targetTable)}\n• 比如：一个用户对应一个用户配置，一个学生对应一个学籍档案\n• 这样设计是为了分离核心数据和扩展数据\n• 便于数据的模块化管理和性能优化`;
        } else if (relType === 'many-to-many') {
            return `🔄 为什么是多对多关系？\n• 举例：多个${this.getTableDisplayName(tableName)}可以关联多个${this.getTableDisplayName(targetTable)}\n• 比如：多个学生可以选择多门课程，多个老师可以教授多个班级\n• 这样设计支持灵活的关联关系和复杂的业务场景\n• 便于实现个性化选择和动态分配`;
        }

        return `🔗 ${tableName} 与 ${targetTable} 之间的 ${relType} 关系\n• 这种关系设计支持数据的结构化组织\n• 便于实现业务逻辑和数据完整性约束\n• 为系统的扩展和维护提供良好的基础`;
    }

    getTableDisplayName(tableName) {
        const displayNames = {
            'users': '用户',
            'schools': '学校',
            'grades': '年级',
            'classes': '班级',
            'subjects': '学科',
            'courses': '课程',
            'course_modules': '课程模块',
            'chapters': '章节',
            'lessons': '课时',
            'homeworks': '作业',
            'questions': '题目',
            'homework_submissions': '作业提交',
            'homework_progress': '作业进度',
            'homework_assignments': '作业分配',
            'class_students': '班级学生',
            'class_schedules': '课程表',
            'user_sessions': '用户会话',
            'notifications': '通知',
            'curriculum_standards': '课程标准'
        };
        return displayNames[tableName] || tableName;
    }

    switchPanelTab(panel, tabName) {
        // 切换标签按钮状态
        panel.querySelectorAll('.panel-tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        panel.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // 切换内容面板
        panel.querySelectorAll('.panel-tab-pane').forEach(pane => {
            pane.classList.remove('active');
        });
        panel.querySelector(`#panel-${tabName}-tab`).classList.add('active');
    }

    async loadPanelTableData(table, panel) {
        console.log(`🔄 开始加载面板表 ${table.name} 的数据...`);

        try {
            const [data, count] = await Promise.all([
                window.databaseAPI.getTableData(table.name, 10, 0),
                window.databaseAPI.getTableCount(table.name)
            ]);

            console.log(`✅ 成功获取面板表 ${table.name} 数据:`, {
                records: data.data?.length || 0,
                total: count
            });

            this.renderPanelTableData(panel, table, data, count);
        } catch (error) {
            console.error(`❌ 加载面板表 ${table.name} 数据失败:`, error);
            this.renderPanelTableDataError(panel, error.message);
        }
    }

    renderPanelTableData(panel, table, data, totalCount) {
        console.log(`🎨 渲染面板表 ${table.name} 数据:`, { data, totalCount });

        const dataContainer = panel.querySelector('.panel-data-container');
        const dataCountElement = panel.querySelector('#panel-data-count');

        if (!data || !data.data) {
            console.warn(`⚠️ 数据格式异常，使用模拟数据`);
            data = window.databaseAPI.getMockData(table.name, 5);
            totalCount = data.total;
        }

        dataCountElement.textContent = `共 ${totalCount} 条记录`;

        if (data.data.length === 0) {
            dataContainer.innerHTML = '<div class="empty-state-mini">该表暂无数据</div>';
            return;
        }

        // 获取前5个字段用于显示
        const fields = table.fields.slice(0, 5);

        dataContainer.innerHTML = `
            <div class="panel-data-table">
                ${data.data.map((row, index) => `
                    <div class="data-row">
                        <div class="row-header">记录 ${index + 1}</div>
                        ${fields.map(field => `
                            <div class="data-field">
                                <span class="field-label">${field.name}:</span>
                                <span class="field-value">${window.databaseAPI.formatValue(row[field.name], field.type)}</span>
                            </div>
                        `).join('')}
                    </div>
                `).join('')}
            </div>
        `;

        console.log(`✅ 面板表 ${table.name} 数据渲染完成`);
    }

    renderPanelTableDataError(panel, errorMessage) {
        const dataContainer = panel.querySelector('.panel-data-container');
        const dataCountElement = panel.querySelector('#panel-data-count');

        dataCountElement.textContent = '数据加载失败';
        dataContainer.innerHTML = `
            <div class="error-state-mini">
                <i class="fas fa-exclamation-triangle"></i>
                <p>数据加载失败: ${errorMessage}</p>
            </div>
        `;
    }

    async refreshPanelData(tableName) {
        const panel = document.querySelector('.table-detail-panel');
        if (!panel) return;

        const table = databaseSchema.tables[tableName];
        if (!table) return;

        // 清除缓存
        window.databaseAPI.clearCache();

        // 显示加载状态
        const dataContainer = panel.querySelector('.panel-data-container');
        dataContainer.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-spinner fa-spin"></i> 刷新数据中...
            </div>
        `;

        // 重新加载数据
        await this.loadPanelTableData(table, panel);
    }








}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    const visualization = new DatabaseVisualization();
    const interactions = new DatabaseInteractions(visualization);

    // 设置全局引用，供模态框使用
    window.interactions = interactions;
    
    // 添加键盘快捷键
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case '=':
                case '+':
                    e.preventDefault();
                    visualization.zoomIn();
                    break;
                case '-':
                    e.preventDefault();
                    visualization.zoomOut();
                    break;
                case '0':
                    e.preventDefault();
                    visualization.resetView();
                    break;
                case 'f':
                    e.preventDefault();
                    document.getElementById('tableSearch').focus();
                    break;
                case 's':
                    e.preventDefault();
                    visualization.exportDiagram();
                    break;
            }
        }
        
        if (e.key === 'Escape') {
            visualization.closeModal();
        }
    });

    // 添加窗口大小变化监听
    window.addEventListener('resize', () => {
        if (visualization.network) {
            visualization.network.redraw();
        }
    });

    // 添加主题切换功能
    const themeToggle = document.createElement('button');
    themeToggle.className = 'theme-toggle btn btn-icon';
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    themeToggle.title = '切换主题';
    document.querySelector('.header-controls').appendChild(themeToggle);

    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        themeToggle.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    });

    // 更新页面统计数据
    const totalTables = Object.keys(databaseSchema.tables).length;
    const totalRelationships = databaseSchema.relationships.length;
    const totalModules = Object.keys(databaseSchema.modules).length;
    const totalFields = Object.values(databaseSchema.tables).reduce((sum, table) => sum + table.fields.length, 0);

    document.getElementById('total-tables').textContent = totalTables;
    document.getElementById('total-fields').textContent = totalFields;
    document.getElementById('total-relationships').textContent = totalRelationships;
    document.getElementById('total-modules').textContent = totalModules;

    console.log('🎉 K-12数学教育系统数据库可视化演示已加载完成！');
    console.log('📊 数据统计:');
    console.log(`   - 数据表: ${totalTables} 张`);
    console.log(`   - 关系连接: ${totalRelationships} 个`);
    console.log(`   - 模块分类: ${totalModules} 个`);
    console.log(`   - 字段总数: ${totalFields} 个`);
    console.log('🔧 快捷键:');
    console.log('   - Ctrl/Cmd + = : 放大');
    console.log('   - Ctrl/Cmd + - : 缩小');
    console.log('   - Ctrl/Cmd + 0 : 重置视图');
    console.log('   - Ctrl/Cmd + F : 搜索');
    console.log('   - Ctrl/Cmd + S : 导出');
    console.log('   - ESC : 关闭模态框');
});
