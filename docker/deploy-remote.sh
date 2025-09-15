#!/bin/bash

# Remote deployment script for K12 Math Education Digital Ecosystem
# Usage: ./deploy-remote.sh

set -e

# Server configuration
SERVER_IP="172.104.172.5"
SERVER_USER="root"
SERVER_PASSWORD="CCNU_rqmWLlqDmx^XF6bOLhF%vSNe*7cYPwk"
PROJECT_NAME="math-ecosystem"
DEPLOY_PATH="/opt/$PROJECT_NAME"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if sshpass is installed
check_sshpass() {
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass is not installed. Installing..."
        # Try to install sshpass
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y sshpass
        elif command -v yum &> /dev/null; then
            sudo yum install -y sshpass
        elif command -v brew &> /dev/null; then
            brew install hudochenkov/sshpass/sshpass
        else
            print_error "Cannot install sshpass automatically. Please install it manually."
            exit 1
        fi
    fi
}

# Execute command on remote server
remote_exec() {
    sshpass -p "$SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SERVER_USER@$SERVER_IP" "$1"
}

# Copy file to remote server
remote_copy() {
    sshpass -p "$SERVER_PASSWORD" scp -o StrictHostKeyChecking=no -r "$1" "$SERVER_USER@$SERVER_IP:$2"
}

# Prepare local files for deployment
prepare_deployment() {
    print_status "Preparing deployment package..."
    
    # Create deployment directory
    rm -rf deploy-package
    mkdir -p deploy-package
    
    # Copy necessary files
    cp docker/Dockerfile deploy-package/
    cp docker/docker-compose.yml deploy-package/
    cp docker/.env.example deploy-package/
    cp docker/deploy.sh deploy-package/
    cp -r docker/nginx deploy-package/
    
    # Copy application code (excluding unnecessary files)
    print_status "Copying application code..."
    
    # Copy backend
    cp -r homework-backend deploy-package/
    
    # Copy frontend source (will be built in Docker)
    cp -r homework_system deploy-package/
    
    # Copy symbol keyboard frontend if exists
    if [ -d "Subject_symbol_dynamic_keyboard/board-frontend" ]; then
        mkdir -p deploy-package/Subject_symbol_dynamic_keyboard
        cp -r Subject_symbol_dynamic_keyboard/board-frontend deploy-package/Subject_symbol_dynamic_keyboard/
    fi
    
    # Copy database schema
    mkdir -p deploy-package/architect
    cp architect/04_数据模型设计_实际版.sql deploy-package/architect/
    
    # Create deployment archive
    tar -czf math-ecosystem-deploy.tar.gz -C deploy-package .
    
    print_success "Deployment package prepared"
}

# Setup remote server
setup_remote_server() {
    print_status "Setting up remote server..."
    
    # Update system and install Docker
    remote_exec "
        apt-get update && 
        apt-get install -y curl wget git &&
        curl -fsSL https://get.docker.com -o get-docker.sh &&
        sh get-docker.sh &&
        systemctl start docker &&
        systemctl enable docker &&
        curl -L \"https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose &&
        chmod +x /usr/local/bin/docker-compose
    "
    
    print_success "Remote server setup completed"
}

# Deploy to remote server
deploy_to_remote() {
    print_status "Deploying to remote server..."
    
    # Create deployment directory
    remote_exec "mkdir -p $DEPLOY_PATH"
    
    # Copy deployment package
    print_status "Uploading deployment package..."
    remote_copy "math-ecosystem-deploy.tar.gz" "$DEPLOY_PATH/"
    
    # Extract and deploy
    remote_exec "
        cd $DEPLOY_PATH &&
        tar -xzf math-ecosystem-deploy.tar.gz &&
        rm math-ecosystem-deploy.tar.gz &&
        chmod +x deploy.sh &&
        cp .env.example .env &&
        ./deploy.sh prod
    "
    
    print_success "Deployment to remote server completed"
}

# Configure firewall
configure_firewall() {
    print_status "Configuring firewall..."
    
    remote_exec "
        ufw allow 22/tcp &&
        ufw allow 80/tcp &&
        ufw allow 443/tcp &&
        ufw --force enable
    "
    
    print_success "Firewall configured"
}

# Setup SSL certificate (Let's Encrypt)
setup_ssl() {
    print_status "Setting up SSL certificate..."
    
    remote_exec "
        apt-get install -y certbot &&
        mkdir -p /var/www/certbot
    "
    
    print_warning "SSL setup prepared. You need to:"
    print_warning "1. Point your domain to $SERVER_IP"
    print_warning "2. Run: certbot certonly --webroot -w /var/www/certbot -d your-domain.com"
    print_warning "3. Update nginx configuration with your domain and SSL paths"
}

# Show deployment information
show_deployment_info() {
    print_success "🎉 Deployment completed successfully!"
    echo ""
    print_status "Server Information:"
    echo "  🖥️  IP Address: $SERVER_IP"
    echo "  👤 Username: $SERVER_USER"
    echo "  📁 Deploy Path: $DEPLOY_PATH"
    echo ""
    print_status "Access URLs:"
    echo "  🌐 Application: http://$SERVER_IP"
    echo "  🔧 API: http://$SERVER_IP/api"
    echo ""
    print_status "Useful SSH Commands:"
    echo "  📋 View logs: sshpass -p '$SERVER_PASSWORD' ssh $SERVER_USER@$SERVER_IP 'cd $DEPLOY_PATH && docker-compose logs -f'"
    echo "  🔄 Restart: sshpass -p '$SERVER_PASSWORD' ssh $SERVER_USER@$SERVER_IP 'cd $DEPLOY_PATH && docker-compose restart'"
    echo "  🛑 Stop: sshpass -p '$SERVER_PASSWORD' ssh $SERVER_USER@$SERVER_IP 'cd $DEPLOY_PATH && docker-compose down'"
}

# Main deployment flow
main() {
    echo "🚀 Remote Deployment for K12 Math Education Digital Ecosystem"
    echo "============================================================="
    
    check_sshpass
    prepare_deployment
    setup_remote_server
    deploy_to_remote
    configure_firewall
    setup_ssl
    show_deployment_info
    
    # Cleanup
    rm -rf deploy-package math-ecosystem-deploy.tar.gz
}

# Run main function
main
