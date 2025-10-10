#!/bin/bash

# GitHub Actions Deployment Script for DigitalOcean
# This script handles deployment from GitHub to the droplet

set -e  # Exit on any error

echo "🚀 Starting GitHub Actions Deployment"
echo "====================================="

# Configuration
APP_DIR="/opt/trading-agent"
SERVICE_NAME="trading-agent"
DB_NAME="options_trading"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if service is running
check_service() {
    if systemctl is-active --quiet $SERVICE_NAME; then
        log "✅ Service $SERVICE_NAME is running"
        return 0
    else
        log "❌ Service $SERVICE_NAME is not running"
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    log "🔍 Checking database connectivity..."
    if sudo -u postgres psql -d $DB_NAME -c "SELECT 1;" > /dev/null 2>&1; then
        log "✅ Database connection successful"
        return 0
    else
        log "❌ Database connection failed"
        return 1
    fi
}

# Main deployment process
main() {
    log "📁 Changing to application directory: $APP_DIR"
    cd $APP_DIR
    
    log "⏹️  Stopping trading agent service..."
    systemctl stop $SERVICE_NAME || log "⚠️  Service was not running"
    
    log "📥 Pulling latest code from GitHub..."
    git pull origin main
    
    log "🐍 Updating Python dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt --quiet
    
    log "🗄️  Updating database schema..."
    python scripts/init_db.py
    
    log "🔄 Starting trading agent service..."
    systemctl start $SERVICE_NAME
    
    # Wait a moment for service to start
    sleep 5
    
    log "🔍 Verifying deployment..."
    
    # Check service status
    if check_service; then
        log "✅ Service started successfully"
    else
        log "❌ Service failed to start"
        systemctl status $SERVICE_NAME --no-pager
        exit 1
    fi
    
    # Check database
    if check_database; then
        log "✅ Database is accessible"
    else
        log "❌ Database check failed"
        exit 1
    fi
    
    # Show final status
    log "📊 Final Status Report:"
    echo "  Service Status: $(systemctl is-active $SERVICE_NAME)"
    echo "  Memory Usage: $(ps aux | grep 'python main.py' | grep -v grep | awk '{print $6/1024 "MB"}' || echo 'N/A')"
    echo "  Data Count: $(sudo -u postgres psql -d $DB_NAME -c "SELECT COUNT(*) FROM index_tick_data;" 2>/dev/null | grep -o '[0-9]*' | tail -1 || echo 'N/A')"
    
    log "🎉 Deployment completed successfully!"
}

# Run main function
main "$@"
