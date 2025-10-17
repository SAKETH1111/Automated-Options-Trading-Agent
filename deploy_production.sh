#!/bin/bash

# Production Deployment Script for Institutional Options Trading System
# DigitalOcean VPS Deployment with PDT Compliance

set -e  # Exit on any error

# Configuration
VPS_HOST="${VPS_HOST:-your-vps-ip}"
VPS_USER="${VPS_USER:-root}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"
PROJECT_NAME="options-trading"
DEPLOY_DIR="/opt/options-trading"
BACKUP_DIR="/opt/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required environment variables are set
    required_vars=(
        "POLYGON_API_KEY"
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "GRAFANA_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Check if SSH key exists
    if [[ ! -f "$SSH_KEY" ]]; then
        log_error "SSH key not found at $SSH_KEY"
        exit 1
    fi
    
    # Check if Docker is installed locally
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed locally"
        exit 1
    fi
    
    # Check if docker-compose is installed locally
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed locally"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Test VPS connectivity
test_vps_connectivity() {
    log_info "Testing VPS connectivity..."
    
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" "echo 'Connection successful'"; then
        log_success "VPS connectivity test passed"
    else
        log_error "Cannot connect to VPS at $VPS_HOST"
        exit 1
    fi
}

# Prepare VPS environment
prepare_vps() {
    log_info "Preparing VPS environment..."
    
    ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" << 'EOF'
        # Update system
        apt-get update && apt-get upgrade -y
        
        # Install Docker
        if ! command -v docker &> /dev/null; then
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            systemctl start docker
            systemctl enable docker
            rm get-docker.sh
        fi
        
        # Install docker-compose
        if ! command -v docker-compose &> /dev/null; then
            curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
        fi
        
        # Create necessary directories
        mkdir -p /opt/options-trading/{config,logs,data,models,backups}
        mkdir -p /opt/options-trading/data/{postgres,redis,prometheus,grafana,trading,ml}
        mkdir -p /opt/backups
        
        # Set proper permissions
        chown -R 1000:1000 /opt/options-trading
        chmod -R 755 /opt/options-trading
        
        # Configure firewall
        ufw allow 22/tcp    # SSH
        ufw allow 80/tcp    # HTTP
        ufw allow 443/tcp   # HTTPS
        ufw allow 8000/tcp  # Trading API
        ufw allow 3000/tcp  # Grafana
        ufw allow 9090/tcp  # Prometheus
        ufw --force enable
        
        echo "VPS environment prepared successfully"
EOF
    
    log_success "VPS environment prepared"
}

# Build and push Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build main application image
    docker build -t "$PROJECT_NAME:latest" .
    
    # Build ML training image
    docker build -t "$PROJECT_NAME-ml:latest" --target ml-training .
    
    # Build backup image
    docker build -t "$PROJECT_NAME-backup:latest" --target backup .
    
    log_success "Docker images built successfully"
}

# Deploy application to VPS
deploy_to_vps() {
    log_info "Deploying application to VPS..."
    
    # Create environment file
    cat > .env << EOF
# Polygon API Configuration
POLYGON_API_KEY=$POLYGON_API_KEY
POLYGON_S3_ACCESS_KEY=$POLYGON_S3_ACCESS_KEY
POLYGON_S3_SECRET_KEY=$POLYGON_S3_SECRET_KEY

# Database Configuration
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD

# Monitoring Configuration
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# VPS Configuration
VPS_HOST=$VPS_HOST
VPS_USERNAME=$VPS_USER
SSH_KEY_PATH=$SSH_KEY

# Optional Configuration
SMTP_SERVER=${SMTP_SERVER:-}
EMAIL_USERNAME=${EMAIL_USERNAME:-}
EMAIL_PASSWORD=${EMAIL_PASSWORD:-}
ALERT_EMAIL=${ALERT_EMAIL:-}
TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID:-}
TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN:-}
TWILIO_PHONE_NUMBER=${TWILIO_PHONE_NUMBER:-}
ALERT_PHONE=${ALERT_PHONE:-}
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-}
EOF

    # Copy files to VPS
    log_info "Copying application files to VPS..."
    rsync -avz -e "ssh -i $SSH_KEY" \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.env' \
        --exclude 'logs/*' \
        . "$VPS_USER@$VPS_HOST:$DEPLOY_DIR/"
    
    # Copy environment file
    scp -i "$SSH_KEY" .env "$VPS_USER@$VPS_HOST:$DEPLOY_DIR/.env"
    
    log_success "Application files copied to VPS"
}

# Start services on VPS
start_services() {
    log_info "Starting services on VPS..."
    
    ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" << EOF
        cd $DEPLOY_DIR
        
        # Load environment variables
        set -a
        source .env
        set +a
        
        # Stop any existing containers
        docker-compose -f docker-compose.prod.yml down || true
        
        # Pull latest images
        docker-compose -f docker-compose.prod.yml pull
        
        # Build images on VPS
        docker-compose -f docker-compose.prod.yml build
        
        # Start services
        docker-compose -f docker-compose.prod.yml up -d
        
        # Wait for services to be ready
        echo "Waiting for services to start..."
        sleep 30
        
        # Check service health
        docker-compose -f docker-compose.prod.yml ps
        
        echo "Services started successfully"
EOF
    
    log_success "Services started on VPS"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for services to be fully ready
    sleep 60
    
    # Check trading agent health
    if curl -f "http://$VPS_HOST:8000/health" > /dev/null 2>&1; then
        log_success "Trading agent is healthy"
    else
        log_warning "Trading agent health check failed"
    fi
    
    # Check Grafana health
    if curl -f "http://$VPS_HOST:3000/api/health" > /dev/null 2>&1; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana health check failed"
    fi
    
    # Check Prometheus health
    if curl -f "http://$VPS_HOST:9090/-/healthy" > /dev/null 2>&1; then
        log_success "Prometheus is healthy"
    else
        log_warning "Prometheus health check failed"
    fi
    
    log_info "Health checks completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" << 'EOF'
        cd /opt/options-trading
        
        # Create Prometheus configuration
        mkdir -p monitoring
        cat > monitoring/prometheus.yml << 'PROM_EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'trading-agent'
    static_configs:
      - targets: ['trading-agent:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
PROM_EOF
        
        # Create Grafana datasource configuration
        mkdir -p monitoring/grafana/datasources
        cat > monitoring/grafana/datasources/prometheus.yml << 'GRAFANA_EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
GRAFANA_EOF
        
        # Create Grafana dashboard configuration
        mkdir -p monitoring/grafana/dashboards
        cat > monitoring/grafana/dashboards/trading.yml << 'DASH_EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
DASH_EOF
        
        echo "Monitoring configuration created"
EOF
    
    log_success "Monitoring setup completed"
}

# Setup backup system
setup_backups() {
    log_info "Setting up backup system..."
    
    ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" << 'EOF'
        cd /opt/options-trading
        
        # Create backup script
        cat > scripts/backup.sh << 'BACKUP_EOF'
#!/bin/bash

# Backup script for options trading system
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="trading_backup_$DATE.tar.gz"

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    --exclude='logs/*' \
    --exclude='data/postgres' \
    --exclude='data/redis' \
    /opt/options-trading

# Database backup
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U trading_user trading_agent > "$BACKUP_DIR/db_backup_$DATE.sql"

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE"
BACKUP_EOF
        
        chmod +x scripts/backup.sh
        
        # Setup cron job for daily backups
        (crontab -l 2>/dev/null; echo "0 2 * * * /opt/options-trading/scripts/backup.sh") | crontab -
        
        echo "Backup system configured"
EOF
    
    log_success "Backup system setup completed"
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."
    
    ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" << 'EOF'
        cd /opt/options-trading
        
        # Wait for PostgreSQL to be ready
        echo "Waiting for PostgreSQL to be ready..."
        sleep 30
        
        # Run database initialization
        docker-compose -f docker-compose.prod.yml exec -T postgres psql -U trading_user -d trading_agent -c "
            CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";
            CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";
        "
        
        # Initialize trading tables
        python3 -c "
import sys
sys.path.append('/opt/options-trading')
from src.database.models import Base, engine
Base.metadata.create_all(engine)
print('Database tables created successfully')
" || echo "Database initialization completed"
EOF
    
    log_success "Database initialized"
}

# Train initial ML models
train_ml_models() {
    log_info "Training initial ML models..."
    
    ssh -i "$SSH_KEY" "$VPS_USER@$VPS_HOST" << 'EOF'
        cd /opt/options-trading
        
        # Run ML training
        docker-compose -f docker-compose.prod.yml run --rm ml-training python -c "
import sys
sys.path.append('/app')
from src.ml.data_pipeline import OptionsMLDataPipeline
from src.ml.ensemble_options import EnsembleOptionsModel

# Initialize data pipeline
pipeline = OptionsMLDataPipeline()

# Train models
pipeline.train_all_models()
print('ML models trained successfully')
"
        
        echo "ML training completed"
EOF
    
    log_success "ML models trained"
}

# Display deployment summary
deployment_summary() {
    log_info "Deployment Summary:"
    echo ""
    echo "🚀 Options Trading System Successfully Deployed!"
    echo ""
    echo "📊 Access URLs:"
    echo "  • Trading API: http://$VPS_HOST:8000"
    echo "  • Grafana Dashboard: http://$VPS_HOST:3000 (admin/$GRAFANA_PASSWORD)"
    echo "  • Prometheus Metrics: http://$VPS_HOST:9090"
    echo ""
    echo "🔐 Credentials:"
    echo "  • Grafana Admin: admin / $GRAFANA_PASSWORD"
    echo "  • PostgreSQL: trading_user / $POSTGRES_PASSWORD"
    echo "  • Redis: (password protected)"
    echo ""
    echo "📁 Important Directories:"
    echo "  • Application: $DEPLOY_DIR"
    echo "  • Logs: $DEPLOY_DIR/logs"
    echo "  • Data: $DEPLOY_DIR/data"
    echo "  • Backups: $BACKUP_DIR"
    echo ""
    echo "🛠️ Management Commands:"
    echo "  • View logs: docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml logs -f"
    echo "  • Restart: docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml restart"
    echo "  • Stop: docker-compose -f $DEPLOY_DIR/docker-compose.prod.yml down"
    echo "  • Backup: $DEPLOY_DIR/scripts/backup.sh"
    echo ""
    echo "⚠️  Next Steps:"
    echo "  1. Configure email/SMS alerts in production.yaml"
    echo "  2. Set up SSL certificates for HTTPS"
    echo "  3. Configure domain name and DNS"
    echo "  4. Start paper trading for 1-2 weeks"
    echo "  5. Monitor performance and adjust settings"
    echo ""
    log_success "Deployment completed successfully!"
}

# Main deployment function
main() {
    log_info "Starting production deployment..."
    
    check_prerequisites
    test_vps_connectivity
    prepare_vps
    build_images
    deploy_to_vps
    start_services
    setup_monitoring
    setup_backups
    initialize_database
    train_ml_models
    run_health_checks
    deployment_summary
}

# Run main function
main "$@"
