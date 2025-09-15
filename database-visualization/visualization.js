// 数据库关系可视化主脚本
class DatabaseVisualization {
    constructor() {
        this.network = null;
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);
        this.currentView = 'overview';
        this.selectedTable = null;
        this.visibleModules = new Set(['user', 'school', 'homework', 'knowledge', 'recommendation']);
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadData();
        this.createNetwork();
        this.hideLoadingIndicator();
    }

    setupEventListeners() {
        // 视图模式切换
        document.getElementById('overviewBtn').addEventListener('click', () => this.switchView('overview'));
        document.getElementById('moduleBtn').addEventListener('click', () => this.switchView('module'));
        document.getElementById('detailBtn').addEventListener('click', () => this.switchView('detail'));

        // 缩放控制
        document.getElementById('zoomIn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoomOut').addEventListener('click', () => this.zoomOut());
        document.getElementById('resetView').addEventListener('click', () => this.resetView());

        // 模块显示控制
        ['user', 'school', 'homework', 'knowledge', 'recommendation'].forEach(module => {
            document.getElementById(`show${module.charAt(0).toUpperCase() + module.slice(1)}`).addEventListener('change', (e) => {
                this.toggleModule(module, e.target.checked);
            });
        });

        // 布局算法切换
        document.getElementById('layoutSelect').addEventListener('change', (e) => {
            this.changeLayout(e.target.value);
        });

        // 搜索功能
        document.getElementById('searchBtn').addEventListener('click', () => this.performSearch());
        document.getElementById('tableSearch').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });

        // 表项点击
        document.querySelectorAll('.table-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const tableName = e.currentTarget.dataset.table;
                this.selectTable(tableName);
            });
        });

        // 分类折叠
        document.querySelectorAll('.category h4').forEach(header => {
            header.addEventListener('click', (e) => {
                const category = e.currentTarget.parentElement;
                category.classList.toggle('collapsed');
            });
        });



        // 导出功能
        document.getElementById('exportBtn').addEventListener('click', () => this.exportDiagram());

        // 全屏功能
        document.getElementById('fullscreenBtn').addEventListener('click', () => this.toggleFullscreen());
    }

    loadData() {
        // 加载节点数据
        const nodeData = [];
        Object.values(databaseSchema.tables).forEach(table => {
            const module = databaseSchema.modules[table.category];
            nodeData.push({
                id: table.name,
                label: table.displayName,
                title: this.createNodeTooltip(table),
                group: table.category,
                color: {
                    background: module.color,
                    border: this.darkenColor(module.color, 20),
                    highlight: {
                        background: this.lightenColor(module.color, 20),
                        border: this.darkenColor(module.color, 40)
                    }
                },
                font: {
                    color: '#333',
                    size: 12,
                    face: 'Arial'
                },
                shape: 'box',
                margin: 10,
                widthConstraint: {
                    minimum: 120,
                    maximum: 200
                }
            });
        });

        // 加载边数据
        const edgeData = [];
        databaseSchema.relationships.forEach((rel, index) => {
            const edgeColor = this.getRelationshipColor(rel.type);
            edgeData.push({
                id: `edge_${index}`,
                from: rel.from,
                to: rel.to,
                label: rel.label || '',
                title: rel.description || `${rel.from} → ${rel.to}`,
                color: {
                    color: edgeColor,
                    highlight: this.darkenColor(edgeColor, 20)
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.8
                    }
                },
                font: {
                    size: 10,
                    color: '#666'
                },
                smooth: {
                    type: 'curvedCW',
                    roundness: 0.2
                },
                width: this.getRelationshipWidth(rel.type)
            });
        });

        this.nodes.update(nodeData);
        this.edges.update(edgeData);
    }

    createNetwork() {
        const container = document.getElementById('networkCanvas');
        const data = {
            nodes: this.nodes,
            edges: this.edges
        };

        const options = {
            layout: {
                hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    nodeSpacing: 200,
                    levelSeparation: 150
                }
            },
            physics: {
                enabled: false
            },
            interaction: {
                dragNodes: true,
                dragView: true,
                zoomView: true,
                selectConnectedEdges: true,
                hover: true
            },
            nodes: {
                borderWidth: 2,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.2)',
                    size: 5,
                    x: 2,
                    y: 2
                }
            },
            edges: {
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 3,
                    x: 1,
                    y: 1
                }
            },
            groups: {
                user: { color: { background: '#ff6b6b' } },
                school: { color: { background: '#4ecdc4' } },
                homework: { color: { background: '#45b7d1' } },
                knowledge: { color: { background: '#96ceb4' } },
                recommendation: { color: { background: '#feca57' } }
            }
        };

        this.network = new vis.Network(container, data, options);

        // 网络事件监听
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                this.selectTable(nodeId);
            }
        });

        this.network.on('doubleClick', (params) => {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                this.showTableDetails(nodeId);
            }
        });

        this.network.on('hoverNode', (params) => {
            this.highlightConnectedNodes(params.node);
        });

        this.network.on('blurNode', () => {
            this.resetHighlight();
        });

        this.network.on('zoom', () => {
            this.updateZoomLevel();
        });
    }

    createNodeTooltip(table) {
        const fieldCount = table.fields.length;
        const relationshipCount = table.relationships.length;
        const module = databaseSchema.modules[table.category];

        // 使用纯文本格式，避免HTML显示问题
        return `${table.displayName}
模块: ${module.name}
描述: ${table.description}
字段数量: ${fieldCount}
关系数量: ${relationshipCount}
键字段: ${table.fields.filter(f => f.isPrimary || f.isForeignKey).length}
💡 双击查看详细信息和实时数据`;
    }

    getRelationshipColor(type) {
        const colors = {
            'one-to-one': '#28a745',
            'one-to-many': '#333333',
            'many-to-many': '#667eea'
        };
        return colors[type] || '#666666';
    }

    getRelationshipWidth(type) {
        const widths = {
            'one-to-one': 3,
            'one-to-many': 2,
            'many-to-many': 2
        };
        return widths[type] || 2;
    }

    switchView(viewType) {
        // 更新按钮状态
        document.querySelectorAll('.view-mode .btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`${viewType}Btn`).classList.add('active');

        this.currentView = viewType;

        // 根据视图类型调整布局
        let layoutOptions = {};
        switch (viewType) {
            case 'overview':
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
            case 'module':
                layoutOptions = {
                    hierarchical: {
                        enabled: false
                    }
                };
                this.arrangeByModules();
                break;
            case 'detail':
                layoutOptions = {
                    hierarchical: {
                        enabled: true,
                        direction: 'LR',
                        sortMethod: 'directed',
                        nodeSpacing: 150,
                        levelSeparation: 200
                    }
                };
                break;
        }

        this.network.setOptions({ layout: layoutOptions });
    }

    arrangeByModules() {
        const positions = {};
        const modulePositions = {
            user: { x: -400, y: -200 },
            school: { x: 0, y: -200 },
            homework: { x: 400, y: -200 },
            knowledge: { x: -200, y: 200 },
            recommendation: { x: 200, y: 200 }
        };

        Object.values(databaseSchema.tables).forEach((table, index) => {
            const modulePos = modulePositions[table.category];
            const tablesInModule = databaseSchema.modules[table.category].tables;
            const tableIndex = tablesInModule.indexOf(table.name);
            const angle = (tableIndex / tablesInModule.length) * 2 * Math.PI;
            const radius = 100;

            positions[table.name] = {
                x: modulePos.x + Math.cos(angle) * radius,
                y: modulePos.y + Math.sin(angle) * radius
            };
        });

        this.network.setData({
            nodes: this.nodes,
            edges: this.edges
        });
        this.network.setOptions({ physics: { enabled: false } });
        this.network.moveNode(positions);
    }

    selectTable(tableName) {
        // 更新选中状态
        document.querySelectorAll('.table-item').forEach(item => {
            item.classList.remove('selected');
        });
        document.querySelector(`[data-table="${tableName}"]`)?.classList.add('selected');

        this.selectedTable = tableName;
        this.updateStatusBar();
        this.showTableDetails(tableName);
        this.highlightTableConnections(tableName);
    }

    showTableDetails(tableName) {
        const table = databaseSchema.tables[tableName];
        if (!table) return;

        // 使用新的右侧面板系统
        if (window.interactions) {
            window.interactions.showTableDetailModal(tableName);
        }
    }



    getFieldIcon(type) {
        if (type.includes('int') || type.includes('decimal')) return 'hashtag';
        if (type.includes('varchar') || type.includes('text')) return 'font';
        if (type.includes('datetime') || type.includes('date')) return 'calendar';
        if (type.includes('json')) return 'code';
        if (type.includes('enum')) return 'list';
        if (type.includes('tinyint')) return 'toggle-on';
        return 'database';
    }

    highlightTableConnections(tableName) {
        // 重置所有节点和边的样式
        this.resetHighlight();

        // 高亮选中的表
        this.nodes.update({
            id: tableName,
            color: {
                background: '#ffd700',
                border: '#ff8c00'
            }
        });

        // 找到相关的边并高亮
        const connectedEdges = this.edges.get().filter(edge => 
            edge.from === tableName || edge.to === tableName
        );

        const connectedNodes = new Set();
        connectedEdges.forEach(edge => {
            // 高亮边
            this.edges.update({
                id: edge.id,
                color: {
                    color: '#ff4757',
                    highlight: '#ff3742'
                },
                width: 4
            });

            // 收集相关节点
            if (edge.from !== tableName) connectedNodes.add(edge.from);
            if (edge.to !== tableName) connectedNodes.add(edge.to);
        });

        // 高亮相关节点
        connectedNodes.forEach(nodeId => {
            const table = databaseSchema.tables[nodeId];
            const module = databaseSchema.modules[table.category];
            this.nodes.update({
                id: nodeId,
                color: {
                    background: this.lightenColor(module.color, 30),
                    border: this.darkenColor(module.color, 20)
                }
            });
        });

        // 更新连接数统计
        document.getElementById('connectionCount').textContent = `连接数: ${connectedEdges.length}`;
    }

    resetHighlight() {
        // 重置所有节点颜色
        Object.values(databaseSchema.tables).forEach(table => {
            const module = databaseSchema.modules[table.category];
            this.nodes.update({
                id: table.name,
                color: {
                    background: module.color,
                    border: this.darkenColor(module.color, 20)
                }
            });
        });

        // 重置所有边颜色
        this.edges.get().forEach(edge => {
            const relationship = databaseSchema.relationships.find(rel => 
                `edge_${databaseSchema.relationships.indexOf(rel)}` === edge.id
            );
            if (relationship) {
                const edgeColor = this.getRelationshipColor(relationship.type);
                this.edges.update({
                    id: edge.id,
                    color: {
                        color: edgeColor,
                        highlight: this.darkenColor(edgeColor, 20)
                    },
                    width: this.getRelationshipWidth(relationship.type)
                });
            }
        });
    }

    // 工具方法
    darkenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    lightenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    hideLoadingIndicator() {
        document.getElementById('loadingIndicator').style.display = 'none';
    }

    updateStatusBar() {
        if (this.selectedTable) {
            const table = databaseSchema.tables[this.selectedTable];
            document.getElementById('selectedTable').textContent = `已选择: ${table.displayName}`;
        } else {
            document.getElementById('selectedTable').textContent = '未选择表';
        }
    }

    updateZoomLevel() {
        const scale = this.network.getScale();
        const percentage = Math.round(scale * 100);
        document.getElementById('zoomLevel').textContent = `缩放: ${percentage}%`;
    }

    // 其他方法将在下一个文件中实现...
}
