// API可视化系统JavaScript
class APIVisualization {
    constructor() {
        this.apiData = null;
        this.baseURL = 'http://172.104.172.5:5001';
        this.init();
    }
    
    async init() {
        try {
            await this.loadAPIData();
            this.renderContent();
        } catch (error) {
            this.showError('加载API数据失败: ' + error.message);
        }
    }
    
    async loadAPIData() {
        try {
            const response = await fetch(`${this.baseURL}/api/apis`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            this.apiData = await response.json();
            console.log('API数据加载成功:', this.apiData);
        } catch (error) {
            console.error('加载API数据失败:', error);
            throw error;
        }
    }
    
    renderContent() {
        if (!this.apiData || !this.apiData.success) {
            this.showError('API数据格式错误');
            return;
        }
        
        // 隐藏加载状态
        document.getElementById('loading').style.display = 'none';
        document.getElementById('content').style.display = 'block';
        
        // 渲染统计信息
        this.renderStats();
        
        // 渲染API分类
        this.renderAPICategories();
        
        // 渲染表与API映射
        this.renderTableAPIMapping();
    }
    
    renderStats() {
        const { apis, categorized_apis } = this.apiData;
        
        // 计算关联的表数量
        const allTables = new Set();
        Object.values(apis).forEach(api => {
            api.database_tables.forEach(table => {
                if (table !== 'dynamic' && table !== 'INFORMATION_SCHEMA.TABLES') {
                    allTables.add(table);
                }
            });
        });
        
        document.getElementById('total-apis').textContent = this.apiData.total_apis;
        document.getElementById('total-categories').textContent = this.apiData.categories.length;
        document.getElementById('total-tables').textContent = allTables.size;
    }
    
    renderAPICategories() {
        const container = document.getElementById('api-categories');
        const { categorized_apis } = this.apiData;
        
        container.innerHTML = '';
        
        // 定义分类图标
        const categoryIcons = {
            'authentication': 'fas fa-user-shield',
            'homework_management': 'fas fa-book',
            'student_features': 'fas fa-user-graduate',
            'grading_system': 'fas fa-check-circle',
            'recommendation_system': 'fas fa-brain',
            'data_visualization': 'fas fa-chart-bar',
            'notification_system': 'fas fa-bell',
            'class_management': 'fas fa-chalkboard-teacher',
            'other': 'fas fa-cog'
        };

        // 定义分类显示名称
        const categoryNames = {
            'authentication': '用户认证',
            'homework_management': '作业管理',
            'student_features': '学生功能',
            'grading_system': '评分系统',
            'recommendation_system': '推荐系统',
            'data_visualization': '数据可视化',
            'notification_system': '通知系统',
            'class_management': '班级管理',
            'other': '其他功能'
        };
        
        Object.entries(categorized_apis).forEach(([category, apis]) => {
            // 跳过没有API的分类
            if (!apis || apis.length === 0) {
                return;
            }

            const categoryCard = document.createElement('div');
            categoryCard.className = 'category-card';

            const icon = categoryIcons[category] || 'fas fa-code';
            const displayName = categoryNames[category] || category;

            categoryCard.innerHTML = `
                <div class="category-header">
                    <div class="category-icon">
                        <i class="${icon}"></i>
                    </div>
                    <h3 class="category-title">${displayName} (${apis.length})</h3>
                </div>
                <ul class="api-list">
                    ${apis.map(api => `
                        <li class="api-item" onclick="apiViz.showAPIDetails('${api.id}')">
                            <div>
                                ${api.methods.map(method =>
                                    `<span class="api-method method-${method}">${method}</span>`
                                ).join('')}
                                <span class="api-path">${api.path}</span>
                            </div>
                            <div class="api-description">${api.description || '暂无描述'}</div>
                        </li>
                    `).join('')}
                </ul>
            `;

            container.appendChild(categoryCard);
        });
    }
    
    async renderTableAPIMapping() {
        const container = document.getElementById('table-list');
        const { apis } = this.apiData;

        try {
            // 从数据库获取所有表
            const tablesResponse = await fetch(`${this.baseURL}/api/database/tables`);
            const tablesData = await tablesResponse.json();

            if (!tablesData.tables) {
                throw new Error('获取数据库表失败');
            }

            // 构建表到API的映射
            const tableAPIMap = {};

            // 初始化所有表
            tablesData.tables.forEach(table => {
                tableAPIMap[table.name] = {
                    count: table.count,
                    apis: []
                };
            });

            // 添加API关联
            Object.values(apis).forEach(api => {
                api.database_tables.forEach(table => {
                    if (table !== 'dynamic' && table !== 'INFORMATION_SCHEMA.TABLES' && tableAPIMap[table]) {
                        tableAPIMap[table].apis.push(api);
                    }
                });
            });

            container.innerHTML = '';

            // 按表名排序
            const sortedTables = Object.entries(tableAPIMap).sort(([a], [b]) => a.localeCompare(b));

            sortedTables.forEach(([tableName, tableInfo]) => {
                const tableItem = document.createElement('div');
                tableItem.className = 'table-item';
                tableItem.onclick = () => this.showTableAPIs(tableName, tableInfo.apis);

                tableItem.innerHTML = `
                    <div class="table-name">
                        <i class="fas fa-table"></i> ${tableName}
                        <span class="badge bg-primary ms-2">${tableInfo.count} 条记录</span>
                        <span class="badge bg-secondary ms-1">${tableInfo.apis.length} APIs</span>
                    </div>
                    <div class="table-apis">
                        ${tableInfo.apis.length > 0 ?
                            tableInfo.apis.slice(0, 5).map(api =>
                                `<span class="table-api-tag">${api.name}</span>`
                            ).join('') :
                            '<span class="text-muted">暂无关联API</span>'
                        }
                        ${tableInfo.apis.length > 5 ? `<span class="table-api-tag">+${tableInfo.apis.length - 5} more</span>` : ''}
                    </div>
                `;

                container.appendChild(tableItem);
            });

        } catch (error) {
            console.error('渲染表API映射失败:', error);
            container.innerHTML = '<p class="text-danger">加载数据表信息失败</p>';
        }
    }
    
    async showAPIDetails(apiId) {
        try {
            const response = await fetch(`${this.baseURL}/api/apis/${apiId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error);
            }
            
            const api = data.api;
            const modalContent = document.getElementById('apiDetailContent');
            
            modalContent.innerHTML = `
                <div class="api-detail-section">
                    <h5 class="api-detail-title">基本信息</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <strong>API名称:</strong> ${api.name}<br>
                            <strong>文件:</strong> ${api.file}<br>
                            <strong>分类:</strong> ${api.category}
                        </div>
                        <div class="col-md-6">
                            <strong>路径:</strong> <code>${api.path}</code><br>
                            <strong>方法:</strong> ${api.methods.map(m => `<span class="api-method method-${m}">${m}</span>`).join(' ')}
                        </div>
                    </div>
                    <div class="mt-2">
                        <strong>描述:</strong> ${api.description || '暂无描述'}
                    </div>
                </div>
                
                <div class="api-detail-section">
                    <h5 class="api-detail-title">请求参数</h5>
                    ${Object.keys(api.parameters).length > 0 ? 
                        Object.entries(api.parameters).map(([name, type]) => `
                            <div class="parameter-item">
                                <span class="parameter-name">${name}</span>
                                <span class="parameter-type">(${type})</span>
                            </div>
                        `).join('') : 
                        '<p class="text-muted">无参数</p>'
                    }
                </div>
                
                <div class="api-detail-section">
                    <h5 class="api-detail-title">响应格式</h5>
                    ${Object.keys(api.responses).length > 0 ? 
                        Object.entries(api.responses).map(([code, fields]) => `
                            <div class="mb-2">
                                <strong>HTTP ${code}:</strong>
                                ${typeof fields === 'object' ? 
                                    Object.entries(fields).map(([name, type]) => `
                                        <div class="response-item">
                                            <span class="response-name">${name}</span>
                                            <span class="response-type">(${type})</span>
                                        </div>
                                    `).join('') : 
                                    `<div class="response-item">${fields}</div>`
                                }
                            </div>
                        `).join('') : 
                        '<p class="text-muted">无响应格式定义</p>'
                    }
                </div>
                
                <div class="api-detail-section">
                    <h5 class="api-detail-title">关联数据表</h5>
                    ${api.database_tables.length > 0 ? `
                        <div class="related-tables">
                            ${api.database_tables.map(table => 
                                `<span class="related-table-tag" onclick="apiViz.showTableDetails('${table}')">${table}</span>`
                            ).join('')}
                        </div>
                    ` : '<p class="text-muted">无关联数据表</p>'}
                </div>
                
                ${Object.keys(api.example_request).length > 0 ? `
                    <div class="api-detail-section">
                        <h5 class="api-detail-title">请求示例</h5>
                        <div class="code-block">
                            <pre>${JSON.stringify(api.example_request, null, 2)}</pre>
                        </div>
                    </div>
                ` : ''}
                
                ${Object.keys(api.example_response).length > 0 ? `
                    <div class="api-detail-section">
                        <h5 class="api-detail-title">响应示例</h5>
                        <div class="code-block">
                            <pre>${JSON.stringify(api.example_response, null, 2)}</pre>
                        </div>
                    </div>
                ` : ''}
            `;
            
            // 显示模态框
            const modal = new bootstrap.Modal(document.getElementById('apiDetailModal'));
            modal.show();
            
        } catch (error) {
            console.error('获取API详情失败:', error);
            alert('获取API详情失败: ' + error.message);
        }
    }
    
    showTableAPIs(tableName, apis) {
        const modalContent = document.getElementById('apiDetailContent');
        
        modalContent.innerHTML = `
            <div class="api-detail-section">
                <h5 class="api-detail-title">数据表: ${tableName}</h5>
                <p class="text-muted">以下API使用了此数据表:</p>
                
                <div class="api-list">
                    ${apis.map(api => `
                        <div class="api-item" onclick="apiViz.showAPIDetails('${api.id}')">
                            <div>
                                ${api.methods.map(method => 
                                    `<span class="api-method method-${method}">${method}</span>`
                                ).join('')}
                                <span class="api-path">${api.path}</span>
                            </div>
                            <div class="api-description">${api.description || '暂无描述'}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        // 显示模态框
        const modal = new bootstrap.Modal(document.getElementById('apiDetailModal'));
        modal.show();

        // 添加模态框关闭事件监听器，确保正确清除body样式
        document.getElementById('apiDetailModal').addEventListener('hidden.bs.modal', function () {
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
            document.body.classList.remove('modal-open');
        });

        // 添加模态框关闭事件监听器，确保正确清除body样式
        document.getElementById('apiDetailModal').addEventListener('hidden.bs.modal', function () {
            document.body.style.overflow = '';
            document.body.style.paddingRight = '';
            document.body.classList.remove('modal-open');
        });
    }
    
    showTableDetails(tableName) {
        // 这里可以集成数据库可视化功能，显示表的详细信息
        console.log('显示表详情:', tableName);
        // 可以跳转到数据库可视化页面或在模态框中显示表结构
        window.open(`index.html?table=${tableName}`, '_blank');
    }
    
    showError(message) {
        document.getElementById('loading').style.display = 'none';
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
}

// 全局实例
let apiViz;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    apiViz = new APIVisualization();
});
