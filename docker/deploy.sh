#!/bin/bash

# K12 Math Education Digital Ecosystem Deployment Script
# Usage: ./deploy.sh [environment]
# Environment: dev, staging, prod (default: dev)

set -e

ENVIRONMENT=${1:-dev}
PROJECT_NAME="math-ecosystem"
COMPOSE_FILE="docker/docker-compose.yml"

echo "🚀 Starting deployment for environment: $ENVIRONMENT"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is installed
check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"
}

# Create environment file if it doesn't exist
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f docker/.env ]; then
        if [ -f docker/.env.example ]; then
            cp docker/.env.example docker/.env
            print_warning "Created docker/.env file from docker/.env.example. Please review and update the configuration."
        else
            print_error "docker/.env.example file not found. Please create environment configuration."
            exit 1
        fi
    fi
    
    # Create necessary directories
    mkdir -p data/uploads logs docker/nginx/ssl
    
    print_success "Environment setup completed"
}

# Build and start services
deploy_services() {
    print_status "Building and starting services..."
    
    # Stop existing services
    docker-compose -f $COMPOSE_FILE down
    
    # Build images
    print_status "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for MySQL
    print_status "Waiting for MySQL to be ready..."
    timeout=60
    while ! docker-compose exec mysql mysqladmin ping -h"localhost" --silent; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            print_error "MySQL failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for Redis
    print_status "Waiting for Redis to be ready..."
    timeout=30
    while ! docker-compose exec redis redis-cli ping; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            print_error "Redis failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
    
    # Wait for application
    print_status "Waiting for application to be ready..."
    timeout=60
    while ! curl -f http://localhost:5000/api/health &> /dev/null; do
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            print_error "Application failed to start within 60 seconds"
            exit 1
        fi
        sleep 1
    done
    
    print_success "All services are ready"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # Check if migration script exists
    if [ -f "homework-backend/scripts/migrate.py" ]; then
        docker-compose exec app python scripts/migrate.py
        print_success "Database migrations completed"
    else
        print_warning "No migration script found, skipping migrations"
    fi
}

# Show deployment status
show_status() {
    print_status "Deployment Status:"
    echo ""
    docker-compose ps
    echo ""
    print_success "🎉 Deployment completed successfully!"
    echo ""
    print_status "Access URLs:"
    echo "  📱 Main Application: http://localhost"
    echo "  🔧 API Documentation: http://localhost/api/docs"
    echo "  📊 Database Visualization: http://localhost/database-visualization"
    echo ""
    print_status "Useful Commands:"
    echo "  📋 View logs: docker-compose logs -f"
    echo "  🔄 Restart services: docker-compose restart"
    echo "  🛑 Stop services: docker-compose down"
    echo "  🗑️  Remove all: docker-compose down -v --remove-orphans"
}

# Main deployment flow
main() {
    echo "🏗️  K12 Math Education Digital Ecosystem Deployment"
    echo "=================================================="
    
    check_docker
    check_docker_compose
    setup_environment
    deploy_services
    wait_for_services
    run_migrations
    show_status
}

# Run main function
main
