#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPA静态文件服务器 - 支持Vue Router的history模式
"""
import os
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

class SPAHandler(SimpleHTTPRequestHandler):
    """支持SPA路由的HTTP处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='dist', **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        # 解析URL
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # 如果是API请求，代理到后端
        if path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        # 检查文件是否存在
        file_path = os.path.join('dist', path.lstrip('/'))
        
        # 如果是目录，添加index.html
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, 'index.html')
        
        # 如果文件不存在，且不是静态资源，返回index.html（SPA路由）
        if not os.path.exists(file_path) and not self.is_static_resource(path):
            file_path = os.path.join('dist', 'index.html')
        
        # 如果文件存在，正常返回
        if os.path.exists(file_path):
            self.serve_file(file_path)
        else:
            self.send_error(404, "File not found")
    
    def is_static_resource(self, path):
        """判断是否为静态资源"""
        static_extensions = ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf']
        return any(path.endswith(ext) for ext in static_extensions)
    
    def serve_file(self, file_path):
        """提供文件服务"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # 设置正确的MIME类型
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type:
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")
    
    def proxy_to_backend(self):
        """代理API请求到后端"""
        import urllib.request
        import urllib.parse
        
        try:
            # 构建后端URL
            backend_url = f"http://172.104.172.5:8081{self.path}"
            
            # 添加查询参数
            if self.path.find('?') != -1:
                backend_url += '&' + self.path.split('?', 1)[1]
            
            # 创建请求
            req = urllib.request.Request(backend_url)
            req.add_header('User-Agent', self.headers.get('User-Agent', ''))
            req.add_header('Authorization', self.headers.get('Authorization', ''))
            
            # 发送请求
            with urllib.request.urlopen(req) as response:
                content = response.read()
                
                # 返回响应
                self.send_response(response.status)
                self.send_header('Content-Type', response.headers.get('Content-Type', 'application/json'))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                self.end_headers()
                self.wfile.write(content)
                
        except Exception as e:
            self.send_error(502, f"Backend proxy error: {e}")
    
    def do_OPTIONS(self):
        """处理CORS预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

def run_server(port=8080):
    """启动SPA服务器"""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, SPAHandler)
    print(f"SPA服务器启动在端口 {port}")
    print(f"访问地址: http://172.104.172.5:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
