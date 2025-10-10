#!/bin/bash
# Automated deployment script for cloud servers

set -e  # Exit on error

echo "========================================="
echo "Trading Agent - Cloud Deployment Script"
echo "========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get server details
read -p "Enter server IP address: " SERVER_IP
read -p "Enter SSH user (default: root): " SSH_USER
SSH_USER=${SSH_USER:-root}

echo ""
echo -e "${YELLOW}Connecting to server...${NC}"

# Test SSH connection
if ! ssh -o ConnectTimeout=5 ${SSH_USER}@${SERVER_IP} "echo 'Connection successful'" > /dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to server${NC}"
    echo "Please check:"
    echo "  - IP address is correct"
    echo "  - SSH key is configured"
    echo "  - Server is running"
    exit 1
fi

echo -e "${GREEN}✅ Connected to server${NC}"

# Copy deployment script to server
echo ""
echo -e "${YELLOW}Copying files to server...${NC}"

cat > /tmp/server_setup.sh << 'EOFSCRIPT'
#!/bin/bash
set -e

echo "========================================="
echo "Setting up server..."
echo "========================================="

# Update system
echo "Updating system packages..."
apt update && apt upgrade -y

# Install dependencies
echo "Installing dependencies..."
apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    postgresql \
    postgresql-contrib \
    nginx \
    certbot

# Create trading user if doesn't exist
if ! id "trader" &>/dev/null; then
    echo "Creating trader user..."
    adduser --disabled-password --gecos "" trader
    usermod -aG sudo trader
    
    # Copy SSH keys
    mkdir -p /home/trader/.ssh
    cp /root/.ssh/authorized_keys /home/trader/.ssh/ 2>/dev/null || true
    chown -R trader:trader /home/trader/.ssh
    chmod 700 /home/trader/.ssh
    chmod 600 /home/trader/.ssh/authorized_keys 2>/dev/null || true
fi

# Setup PostgreSQL database
echo "Setting up database..."
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'trading_agent'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE DATABASE trading_agent;"

sudo -u postgres psql -tc "SELECT 1 FROM pg_user WHERE usename = 'trader'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER trader WITH PASSWORD 'trading_secure_pass_123';"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_agent TO trader;"

# Setup firewall
echo "Configuring firewall..."
ufw --force enable
ufw allow 22/tcp  # SSH
ufw allow 80/tcp  # HTTP (for monitoring)
ufw allow 443/tcp # HTTPS

echo "✅ Server setup complete!"
EOFSCRIPT

# Upload and run setup script
scp /tmp/server_setup.sh ${SSH_USER}@${SERVER_IP}:/tmp/
ssh ${SSH_USER}@${SERVER_IP} "chmod +x /tmp/server_setup.sh && sudo /tmp/server_setup.sh"

echo -e "${GREEN}✅ Server configured${NC}"

# Clone repository
echo ""
echo -e "${YELLOW}Deploying application...${NC}"

ssh ${SSH_USER}@${SERVER_IP} << 'EOFAPP'
su - trader << 'EOFTRADER'
set -e

# Clone repository if not exists
if [ ! -d "Automated-Options-Trading-Agent" ]; then
    echo "Cloning repository..."
    git clone https://github.com/YOUR_USERNAME/Automated-Options-Trading-Agent.git
fi

cd Automated-Options-Trading-Agent

# Pull latest changes
echo "Pulling latest code..."
git pull origin main || true

# Setup Python environment
echo "Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python scripts/init_db.py || true
python scripts/migrate_add_tick_data.py || true

echo "✅ Application deployed!"
EOFTRADER
EOFAPP

echo -e "${GREEN}✅ Application deployed${NC}"

# Create systemd service
echo ""
echo -e "${YELLOW}Creating systemd service...${NC}"

ssh ${SSH_USER}@${SERVER_IP} << 'EOFSYSTEMD'
cat > /tmp/trading-agent.service << 'EOFSERVICE'
[Unit]
Description=Automated Trading Agent
After=network.target postgresql.service

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/Automated-Options-Trading-Agent
Environment="PATH=/home/trader/Automated-Options-Trading-Agent/venv/bin"
ExecStart=/home/trader/Automated-Options-Trading-Agent/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/trader/Automated-Options-Trading-Agent/logs/systemd.log
StandardError=append:/home/trader/Automated-Options-Trading-Agent/logs/systemd.log

[Install]
WantedBy=multi-user.target
EOFSERVICE

sudo mv /tmp/trading-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-agent
EOFSYSTEMD

echo -e "${GREEN}✅ Systemd service created${NC}"

# Prompt for environment variables
echo ""
echo -e "${YELLOW}Configuration needed:${NC}"
echo "The agent requires API keys and configuration."
echo ""
echo "Please SSH to the server and:"
echo "  1. Edit /home/trader/Automated-Options-Trading-Agent/.env"
echo "  2. Add your Alpaca API keys"
echo "  3. Start the service: sudo systemctl start trading-agent"
echo ""
echo "SSH command: ssh trader@${SERVER_IP}"
echo ""

read -p "Do you want to configure now? (y/n): " CONFIGURE_NOW

if [ "$CONFIGURE_NOW" = "y" ]; then
    echo ""
    echo "Opening SSH session..."
    echo "After editing .env, run: sudo systemctl start trading-agent"
    ssh trader@${SERVER_IP}
fi

echo ""
echo "========================================="
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "========================================="
echo ""
echo "Server IP: ${SERVER_IP}"
echo "SSH: ssh trader@${SERVER_IP}"
echo ""
echo "Next steps:"
echo "  1. Configure .env with API keys"
echo "  2. Start service: sudo systemctl start trading-agent"
echo "  3. Check status: sudo systemctl status trading-agent"
echo "  4. View logs: tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log"
echo ""
echo "Monitoring:"
echo "  - Health check: python scripts/system_health.py"
echo "  - View logs: tail -f logs/trading_agent.log"
echo ""

