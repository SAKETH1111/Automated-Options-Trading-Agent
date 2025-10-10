# ☁️ Cloud Deployment Guide - Run 24/7 Without Your Laptop

## Overview

Deploy your trading agent to the cloud so it runs **24/7 independently** of your laptop. This guide covers multiple deployment options from beginner-friendly to advanced.

## Deployment Options

### 1. **DigitalOcean Droplet** 💧 (Recommended for Beginners)
- **Cost**: $6-12/month
- **Difficulty**: Easy
- **Setup Time**: 15 minutes
- **Best For**: Simple, reliable, affordable

### 2. **AWS EC2** ☁️ (Most Popular)
- **Cost**: $10-20/month (t3.micro)
- **Difficulty**: Medium
- **Setup Time**: 30 minutes
- **Best For**: Scalability, AWS ecosystem

### 3. **Google Cloud VM** 🌐
- **Cost**: $8-15/month
- **Difficulty**: Medium
- **Setup Time**: 30 minutes
- **Best For**: Google services integration

### 4. **Raspberry Pi** 🥧 (Self-Hosted)
- **Cost**: $50 one-time + electricity
- **Difficulty**: Medium
- **Setup Time**: 1 hour
- **Best For**: Learning, full control

### 5. **Docker Container** 🐳 (Any Platform)
- **Cost**: Varies by hosting
- **Difficulty**: Medium-Hard
- **Setup Time**: 45 minutes
- **Best For**: Portability, isolation

---

## Quick Start: DigitalOcean Deployment (Easiest)

### Step 1: Create Droplet (5 minutes)

1. Sign up at [digitalocean.com](https://digitalocean.com)
2. Create new Droplet:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic ($6/month) or Regular ($12/month)
   - **Region**: Choose closest to you
   - **Authentication**: SSH key (recommended)

3. Note your droplet's IP address (e.g., 167.99.123.45)

### Step 2: Connect to Server (2 minutes)

```bash
# From your laptop
ssh root@YOUR_DROPLET_IP

# Example:
ssh root@167.99.123.45
```

### Step 3: Setup Server (5 minutes)

```bash
# Update system
apt update && apt upgrade -y

# Install Python 3.11
apt install python3.11 python3.11-venv python3-pip git -y

# Install PostgreSQL (for database)
apt install postgresql postgresql-contrib -y

# Create trading user
adduser trader
usermod -aG sudo trader
su - trader
```

### Step 4: Deploy Your Code (5 minutes)

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/Automated-Options-Trading-Agent.git
cd Automated-Options-Trading-Agent

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
nano .env
# Paste your API keys, etc.
```

### Step 5: Setup Database (3 minutes)

```bash
# Setup PostgreSQL
sudo -u postgres psql

# In PostgreSQL:
CREATE DATABASE trading_agent;
CREATE USER trader WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE trading_agent TO trader;
\q

# Run migrations
python scripts/init_db.py
python scripts/migrate_add_tick_data.py
```

### Step 6: Run Agent as Service (5 minutes)

Create systemd service:

```bash
sudo nano /etc/systemd/system/trading-agent.service
```

Paste:
```ini
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

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-agent
sudo systemctl start trading-agent

# Check status
sudo systemctl status trading-agent

# View logs
sudo journalctl -u trading-agent -f
```

### Step 7: Monitor (Ongoing)

```bash
# Check if running
sudo systemctl status trading-agent

# View logs
tail -f logs/trading_agent.log

# Check health
python scripts/system_health.py
```

---

## Complete AWS EC2 Deployment

### 1. Launch EC2 Instance

**AWS Console**:
1. Go to EC2 Dashboard
2. Click "Launch Instance"
3. Choose:
   - **AMI**: Ubuntu Server 22.04 LTS
   - **Instance Type**: t3.micro (free tier) or t3.small
   - **Storage**: 20 GB
   - **Security Group**: 
     - Allow SSH (port 22) from your IP
     - Optional: Allow port 8000 for monitoring dashboard

4. Create/select key pair
5. Launch instance

### 2. Connect to Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### 3. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip git postgresql -y

# Setup database
sudo -u postgres createdb trading_agent
sudo -u postgres psql -c "CREATE USER trader WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_agent TO trader;"
```

### 4. Deploy Application

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/Automated-Options-Trading-Agent.git
cd Automated-Options-Trading-Agent

# Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Add your API keys

# Initialize database
python scripts/init_db.py
python scripts/migrate_add_tick_data.py
```

### 5. Create Systemd Service

Same as DigitalOcean (Step 6 above)

### 6. Setup Monitoring

```bash
# Install monitoring tools
pip install psutil

# Create monitoring script
nano scripts/monitor_server.py
```

---

## Docker Deployment (Advanced)

### Why Docker?
- ✅ Isolated environment
- ✅ Easy to move between servers
- ✅ Consistent deployment
- ✅ Easy updates

### 1. Create Dockerfile

Already created! Use the existing `Dockerfile` in your repo.

### 2. Build Image

```bash
docker build -t trading-agent:latest .
```

### 3. Run Container

```bash
docker run -d \
  --name trading-agent \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/.env:/app/.env \
  trading-agent:latest
```

### 4. View Logs

```bash
docker logs -f trading-agent
```

### 5. Deploy to Cloud with Docker

**Option A: AWS ECS (Elastic Container Service)**
- Managed container service
- Auto-scaling
- $20-40/month

**Option B: Google Cloud Run**
- Serverless containers
- Pay per use
- $10-30/month

**Option C: DigitalOcean App Platform**
- Managed containers
- Easy deployment
- $12-25/month

---

## Raspberry Pi Deployment (Self-Hosted)

### Hardware Needed
- Raspberry Pi 4 (4GB+ RAM): $55
- MicroSD Card (32GB): $10
- Power Supply: $8
- Case: $5
- **Total**: ~$80

### Setup Steps

1. **Install Raspberry Pi OS**
   - Download Raspberry Pi Imager
   - Flash OS to SD card
   - Enable SSH

2. **Boot and Connect**
   ```bash
   ssh pi@raspberrypi.local
   # Default password: raspberry
   ```

3. **Update System**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3-pip python3-venv git -y
   ```

4. **Deploy Application**
   Same as DigitalOcean (Steps 4-6)

5. **Keep Pi Running**
   - Use UPS for power backup
   - Enable auto-start on boot
   - Monitor temperature

---

## Cost Comparison

| Option | Monthly Cost | Setup | Reliability |
|--------|--------------|-------|-------------|
| DigitalOcean | $6-12 | Easy | High |
| AWS EC2 | $10-20 | Medium | Very High |
| Google Cloud | $8-15 | Medium | Very High |
| Raspberry Pi | $2-3 (electricity) | Medium | Medium |
| Docker (managed) | $12-40 | Hard | Very High |

**Recommended**: Start with **DigitalOcean $6/month** droplet.

---

## Essential Configuration

### 1. Environment Variables

Make sure `.env` has:
```bash
# Alpaca API
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database (if using PostgreSQL)
DATABASE_URL=postgresql://trader:password@localhost/trading_agent

# Monitoring
ALERT_EMAIL=your@email.com
```

### 2. Update Configuration

Edit `config/spy_qqq_config.yaml`:
```yaml
trading:
  mode: paper  # or live when ready
  
monitoring:
  alerts:
    enabled: true
    channels:
      - console
      - email  # or webhook
```

### 3. Setup Automatic Updates

```bash
# Create update script
nano scripts/update_system.sh
```

```bash
#!/bin/bash
cd /home/trader/Automated-Options-Trading-Agent
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart trading-agent
```

```bash
chmod +x scripts/update_system.sh

# Schedule weekly updates (Sunday 2 AM)
crontab -e
# Add:
0 2 * * 0 /home/trader/Automated-Options-Trading-Agent/scripts/update_system.sh
```

---

## Monitoring & Alerts

### 1. Setup Email Alerts

Install:
```bash
pip install sendgrid  # or use SMTP
```

Configure in `src/monitoring/alerts.py`

### 2. Setup Uptime Monitoring

**Option A: UptimeRobot (Free)**
- Monitor HTTP endpoint every 5 minutes
- Email/SMS alerts if down

**Option B: Healthchecks.io (Free)**
- Ping endpoint every minute
- Alerts if no ping received

### 3. Setup Log Monitoring

**Logtail** (Free tier):
```bash
pip install logtail-python

# Add to your code:
from logtail import LogtailHandler
import logging

logger = logging.getLogger()
handler = LogtailHandler(source_token="YOUR_TOKEN")
logger.addHandler(handler)
```

### 4. Setup Dashboard

**Grafana + Prometheus**:
```bash
# Install Prometheus
sudo apt install prometheus -y

# Install Grafana
sudo apt install grafana -y

# Access at http://YOUR_IP:3000
```

---

## Security Best Practices

### 1. Firewall Setup

```bash
# Enable UFW firewall
sudo ufw allow 22/tcp  # SSH
sudo ufw enable

# If using monitoring dashboard:
sudo ufw allow 8000/tcp
```

### 2. SSH Security

```bash
# Disable password authentication
sudo nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

### 3. Automatic Security Updates

```bash
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 4. Backup Strategy

```bash
# Create backup script
nano scripts/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/trader/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump trading_agent > $BACKUP_DIR/db_$DATE.sql

# Backup data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz data/ logs/

# Keep only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

```bash
chmod +x scripts/backup.sh

# Run daily at 1 AM
crontab -e
# Add:
0 1 * * * /home/trader/Automated-Options-Trading-Agent/scripts/backup.sh
```

---

## Troubleshooting

### Agent Won't Start

```bash
# Check service status
sudo systemctl status trading-agent

# View errors
sudo journalctl -u trading-agent -n 50

# Check Python errors
cd /home/trader/Automated-Options-Trading-Agent
source venv/bin/activate
python main.py  # Run manually to see errors
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U trader -d trading_agent -h localhost
```

### Out of Memory

```bash
# Check memory usage
free -h

# Increase swap if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### High CPU Usage

```bash
# Check processes
top

# Reduce collection interval
# Edit config/spy_qqq_config.yaml:
# collect_interval_seconds: 5.0  # Instead of 1.0
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] Test locally thoroughly
- [ ] Configure all API keys
- [ ] Set up alerts/monitoring
- [ ] Choose cloud provider
- [ ] Estimate costs

### Deployment
- [ ] Create server/instance
- [ ] Install dependencies
- [ ] Clone repository
- [ ] Configure environment
- [ ] Setup database
- [ ] Run migrations
- [ ] Create systemd service
- [ ] Test manually first

### Post-Deployment
- [ ] Verify agent is running
- [ ] Check logs for errors
- [ ] Monitor for 24 hours
- [ ] Setup backups
- [ ] Configure alerts
- [ ] Document access details
- [ ] Test failure scenarios

---

## Cost Optimization

### 1. Use Reserved Instances
AWS reserved instances save 30-60%

### 2. Stop During Weekends
Market closed = no need to run

```bash
# Stop Friday 5 PM
0 17 * * 5 sudo systemctl stop trading-agent

# Start Monday 9 AM
0 9 * * 1 sudo systemctl start trading-agent
```

### 3. Use Spot Instances
AWS/GCP spot instances are 70% cheaper (but can be interrupted)

### 4. Optimize Resource Usage
- Use t3.micro instead of t3.small
- Store logs in S3 after 7 days
- Use managed databases only if needed

---

## Next Steps

### Quick Start (15 minutes)
1. ✅ Choose: **DigitalOcean $6/month droplet**
2. ✅ Follow: Steps 1-6 in "Quick Start" section
3. ✅ Monitor: Check logs and health

### Production Ready (1 hour)
1. ✅ Setup monitoring (UptimeRobot)
2. ✅ Configure email alerts
3. ✅ Setup daily backups
4. ✅ Enable auto-updates
5. ✅ Document everything

### Advanced (Optional)
1. Setup Docker deployment
2. Add Grafana dashboard
3. Implement CI/CD pipeline
4. Multi-region deployment

---

## Summary

**To run without your laptop**:
1. ✅ Deploy to cloud server ($6-20/month)
2. ✅ Run as background service
3. ✅ Setup monitoring & alerts
4. ✅ Configure automatic backups

**Recommended path**:
- **Beginners**: DigitalOcean Droplet ($6/month)
- **AWS Users**: EC2 t3.micro ($10/month)
- **Advanced**: Docker on managed platform
- **Self-hosted**: Raspberry Pi at home

**Your agent will run 24/7** independently! 🚀

---

## Getting Help

- **DigitalOcean**: Extensive tutorials and community
- **AWS**: Free tier for 12 months
- **Docker**: Official documentation
- **This guide**: Step-by-step instructions

Start with the "Quick Start: DigitalOcean" section above! ☁️

