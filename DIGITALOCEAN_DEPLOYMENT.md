# 🌊 DigitalOcean Deployment - Step-by-Step Guide

## Let's Deploy Your Trading Agent to DigitalOcean!

This guide will walk you through deploying your trading agent to a DigitalOcean Droplet so it runs 24/7.

**Time needed**: 20-30 minutes  
**Cost**: $6/month  
**Difficulty**: Easy ⭐☆☆

---

## Step 1: Create DigitalOcean Account (5 minutes)

### 1.1 Sign Up

1. Go to **[digitalocean.com](https://digitalocean.com)**
2. Click **"Sign Up"**
3. Create account with:
   - Email address
   - Password
   - Or sign up with Google/GitHub

### 1.2 Add Payment Method

1. Go to **Billing** in left sidebar
2. Add credit card or PayPal
3. You'll be charged $6/month ($0.20/day) for the droplet

### 1.3 Verify Email

Check your email and click verification link.

---

## Step 2: Create Your Droplet (5 minutes)

### 2.1 Start Creating

1. Click **"Create"** (green button, top right)
2. Select **"Droplets"**

### 2.2 Choose Configuration

**Region:**
- Choose closest to you
- Recommended: **New York** or **San Francisco**

**Image:**
- Click **"Marketplace"** tab
- Search for **"Docker"** (optional, or use Ubuntu)
- Or just use: **Ubuntu 22.04 LTS** ✅

**Droplet Size:**

Click **"Basic"** plan:

| Plan | RAM | CPU | Storage | Price |
|------|-----|-----|---------|-------|
| Regular | 1 GB | 1 vCPU | 25 GB | **$6/month** ✅ |
| Regular | 2 GB | 1 vCPU | 50 GB | $12/month |

**Choose**: $6/month (Regular, 1GB RAM) - Perfect for starting!

**Authentication:**

Choose **"Password"** (easier) or **"SSH Key"** (more secure)

If choosing password:
- Create a strong password
- Save it securely

If choosing SSH Key:
- Click **"New SSH Key"**
- Follow instructions to add your laptop's SSH key

**Hostname:**
- Give it a name like: `trading-agent-01`

### 2.3 Create Droplet

1. Click **"Create Droplet"** (green button at bottom)
2. Wait 1-2 minutes for droplet to be created
3. **Copy your Droplet's IP address** (e.g., 167.99.123.45)
   - Save this! You'll need it

---

## Step 3: Prepare for Deployment (2 minutes)

### 3.1 Test Connection

Open Terminal on your laptop:

```bash
# Test if droplet is reachable (replace with your IP)
ping 167.99.123.45

# Press Ctrl+C to stop after a few pings
```

If you see replies, you're good! ✅

### 3.2 First SSH Connection

```bash
# Connect to your droplet (replace with your IP)
ssh root@167.99.123.45

# If using password, enter the password you set
# If prompted about fingerprint, type 'yes'
```

You should now be connected to your droplet! 🎉

```bash
# You'll see something like:
root@trading-agent-01:~#
```

**Don't close this terminal yet!** Or if you do, that's fine - we can reconnect.

---

## Step 4: Deploy Using Automated Script (10 minutes)

### Option A: Use Our Automated Script (Recommended)

**On your laptop** (open a new terminal, don't close the server one):

```bash
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent

# Run deployment script
./scripts/deploy_to_server.sh
```

**When prompted:**
```
Enter server IP address: 167.99.123.45  # Your droplet IP
Enter SSH user: root                     # Default user
```

The script will automatically:
- ✅ Update system
- ✅ Install Python 3.11, PostgreSQL
- ✅ Create trader user
- ✅ Setup database
- ✅ Deploy your code
- ✅ Create systemd service

**Wait 5-10 minutes** while it installs everything.

### Option B: Manual Deployment (If Script Doesn't Work)

If the automated script has issues, follow the manual steps in the next section.

---

## Step 5: Configure Your Agent (5 minutes)

After deployment completes, you need to add your Alpaca API keys.

### 5.1 SSH to Droplet

```bash
# If not already connected
ssh root@167.99.123.45

# Switch to trader user
su - trader
```

### 5.2 Navigate to App Directory

```bash
cd Automated-Options-Trading-Agent
```

### 5.3 Create .env File

```bash
nano .env
```

### 5.4 Add Your Configuration

Paste this (replace with YOUR actual keys):

```bash
# Alpaca API Keys
ALPACA_API_KEY=PKxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database
DATABASE_URL=postgresql://trader:trading_secure_pass_123@localhost/trading_agent

# Optional: Alerts
ALERT_EMAIL=your@email.com
```

**Important**: 
- Replace `PKxxxxxxxxxxxx` with your actual Alpaca API key
- Replace the secret key with your actual secret
- Use `paper-api.alpaca.markets` for paper trading
- Use `api.alpaca.markets` for live trading (only when ready!)

### 5.5 Save the File

- Press `Ctrl + X`
- Press `Y` to confirm
- Press `Enter` to save

---

## Step 6: Start Your Trading Agent (2 minutes)

### 6.1 Initialize Database

```bash
# Make sure you're in the app directory
cd ~/Automated-Options-Trading-Agent

# Activate virtual environment
source venv/bin/activate

# Initialize database
python scripts/init_db.py
python scripts/migrate_add_tick_data.py
```

You should see:
```
✅ Database initialized successfully
✅ Migration completed successfully
```

### 6.2 Start the Service

```bash
# Start the trading agent
sudo systemctl start trading-agent

# Check if it's running
sudo systemctl status trading-agent
```

You should see:
```
● trading-agent.service - Automated Trading Agent
   Active: active (running) since ...
```

Press `q` to exit the status view.

### 6.3 Watch the Logs

```bash
# View real-time logs
tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log
```

You should start seeing logs like:
```
2025-01-10 14:30:45 | INFO | Trading Orchestrator initialized
2025-01-10 14:30:45 | INFO | RealTimeDataCollector initialized for ['SPY', 'QQQ']
2025-01-10 14:30:46 | INFO | ✅ Real-time data collection started
2025-01-10 14:30:46 | INFO | ✅ Trading Agent is now LIVE
```

**Congratulations! 🎉 Your agent is now running 24/7!**

Press `Ctrl + C` to stop viewing logs (agent keeps running).

---

## Step 7: Verify Everything is Working (5 minutes)

### 7.1 Check System Health

```bash
cd ~/Automated-Options-Trading-Agent
python scripts/system_health.py
```

You should see:
```
✅ Overall Status: HEALTHY

✅ RobustDataCollector: healthy
  Operations:    45
  Success Rate:  100.0%
  Data Age:      2s
```

### 7.2 Check Service Status

```bash
sudo systemctl status trading-agent
```

Should show: `Active: active (running)`

### 7.3 Check Data Collection

```bash
# Wait a few minutes, then check if data is being collected
cd ~/Automated-Options-Trading-Agent
source venv/bin/activate
python -c "
from src.market_data.tick_analyzer import TickDataAnalyzer
analyzer = TickDataAnalyzer()
avail = analyzer.get_data_availability('SPY', days=1)
print(f'Total ticks collected: {avail.get(\"total_ticks\", 0)}')
"
```

You should see ticks being collected (if market is open).

---

## Step 8: Monitor from Your Laptop (Ongoing)

### 8.1 Create an Alias (Optional but Convenient)

On **your laptop**, add this to `~/.zshrc` or `~/.bashrc`:

```bash
# Add to end of file
alias trading-ssh='ssh trader@167.99.123.45'
alias trading-logs='ssh trader@167.99.123.45 "tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log"'
alias trading-status='ssh trader@167.99.123.45 "sudo systemctl status trading-agent"'
alias trading-health='./scripts/check_remote_health.sh 167.99.123.45'
```

Then reload:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

Now you can use:
```bash
trading-ssh      # SSH to your droplet
trading-logs     # View logs
trading-status   # Check status
trading-health   # Full health check
```

### 8.2 Quick Health Check from Laptop

```bash
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent
./scripts/check_remote_health.sh 167.99.123.45
```

### 8.3 View Logs from Laptop

```bash
ssh trader@167.99.123.45 "tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log"
```

---

## Common Commands Reference

### Control the Agent

```bash
# Start
sudo systemctl start trading-agent

# Stop
sudo systemctl stop trading-agent

# Restart
sudo systemctl restart trading-agent

# Check status
sudo systemctl status trading-agent

# Enable auto-start on boot (already done)
sudo systemctl enable trading-agent
```

### View Logs

```bash
# Real-time logs
tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log

# Last 100 lines
tail -n 100 ~/Automated-Options-Trading-Agent/logs/trading_agent.log

# System logs
sudo journalctl -u trading-agent -f

# Last 50 system logs
sudo journalctl -u trading-agent -n 50
```

### Update Code

```bash
# SSH to droplet
ssh trader@167.99.123.45

# Pull latest changes
cd ~/Automated-Options-Trading-Agent
git pull origin main

# Restart agent
sudo systemctl restart trading-agent
```

### Check Data Collection

```bash
cd ~/Automated-Options-Trading-Agent
source venv/bin/activate
python scripts/view_tick_data.py
```

---

## Troubleshooting

### Issue: Agent Won't Start

**Check logs:**
```bash
sudo journalctl -u trading-agent -n 100
```

**Common fixes:**

1. **Missing API keys**
   ```bash
   nano ~/Automated-Options-Trading-Agent/.env
   # Add your keys
   sudo systemctl restart trading-agent
   ```

2. **Database not initialized**
   ```bash
   cd ~/Automated-Options-Trading-Agent
   source venv/bin/activate
   python scripts/init_db.py
   sudo systemctl restart trading-agent
   ```

3. **Python dependencies missing**
   ```bash
   cd ~/Automated-Options-Trading-Agent
   source venv/bin/activate
   pip install -r requirements.txt
   sudo systemctl restart trading-agent
   ```

### Issue: Can't SSH to Droplet

**Solutions:**

1. Check droplet is running in DigitalOcean console
2. Verify IP address is correct
3. Check password/SSH key
4. Try from DigitalOcean console "Access" → "Launch Droplet Console"

### Issue: Out of Memory

```bash
# Check memory
free -h

# Add swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Issue: High CPU Usage

If running on $6/month droplet, reduce collection frequency:

```bash
nano ~/Automated-Options-Trading-Agent/config/spy_qqq_config.yaml

# Change:
collect_interval_seconds: 5.0  # Instead of 1.0
```

---

## Cost Optimization

### Stop During Weekends

Market is closed on weekends, so you can stop the agent:

```bash
# Add to crontab
crontab -e

# Add these lines:
0 17 * * 5 sudo systemctl stop trading-agent  # Friday 5 PM EST
0 9 * * 1 sudo systemctl start trading-agent  # Monday 9 AM EST
```

### Upgrade Only if Needed

Start with $6/month. Only upgrade if:
- Running out of memory consistently
- High CPU usage affecting performance
- Need faster collection (every second vs every 5 seconds)

---

## Security Best Practices

### 1. Setup SSH Keys (Highly Recommended)

**On your laptop:**
```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your@email.com"

# Copy to droplet
ssh-copy-id trader@167.99.123.45
```

### 2. Disable Password Authentication

**On droplet:**
```bash
sudo nano /etc/ssh/sshd_config

# Find and change:
PasswordAuthentication no

# Save and restart SSH
sudo systemctl restart sshd
```

### 3. Enable Firewall

```bash
sudo ufw allow 22/tcp  # SSH
sudo ufw enable
```

### 4. Setup Automatic Security Updates

```bash
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Monitoring & Alerts

### Setup UptimeRobot (Free)

1. Go to [uptimerobot.com](https://uptimerobot.com)
2. Sign up (free account)
3. Add monitor:
   - Type: Ping
   - Target: Your droplet IP
   - Interval: Every 5 minutes
4. Add email alert

You'll get notified if server goes down!

### Setup HealthChecks.io (Free)

1. Go to [healthchecks.io](https://healthchecks.io)
2. Create account
3. Create check: "Trading Agent"
4. Add this to your agent to ping every 5 minutes

---

## Backup Strategy

### Manual Backup

```bash
# On droplet
cd ~/Automated-Options-Trading-Agent
tar -czf ~/backup_$(date +%Y%m%d).tar.gz data/ logs/ .env

# Download to laptop
# On laptop:
scp trader@167.99.123.45:~/backup_*.tar.gz ~/Downloads/
```

### Automated Daily Backups

```bash
# Create backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR=~/backups
mkdir -p $BACKUP_DIR
DATE=$(date +%Y%m%d)

cd ~/Automated-Options-Trading-Agent
tar -czf $BACKUP_DIR/trading_backup_$DATE.tar.gz data/ logs/ .env

# Keep only last 7 days
find $BACKUP_DIR -name "trading_backup_*.tar.gz" -mtime +7 -delete
```

```bash
chmod +x ~/backup.sh

# Schedule daily at 2 AM
crontab -e
# Add:
0 2 * * * /home/trader/backup.sh
```

---

## What's Running on Your Droplet

### Services:
- ✅ PostgreSQL database
- ✅ Trading agent (Python)
- ✅ Systemd service manager

### Ports:
- 22: SSH (open)
- 5432: PostgreSQL (internal only)

### Resources:
- RAM: ~300-500 MB used
- CPU: 5-15% average
- Disk: ~2-5 GB used

---

## Next Steps

### After 24 Hours

1. **Check logs** for any errors
2. **Verify data collection** is working
3. **Monitor resource usage**
4. **Review collected tick data**

### After 1 Week

1. **Analyze performance** metrics
2. **Check error rates**
3. **Optimize if needed**
4. **Consider upgrading** to $12/month if needed

### When Ready for Live Trading

1. **Test thoroughly** on paper trading (1-2 weeks minimum)
2. **Review all logs** for errors
3. **Verify strategies** are working as expected
4. **Change** `ALPACA_BASE_URL` to live API
5. **Start with small positions**

---

## Summary

**You now have:**
- ✅ Trading agent running 24/7 on DigitalOcean
- ✅ $6/month cost
- ✅ Automatic restarts if crashes
- ✅ Data being collected every second
- ✅ Health monitoring active
- ✅ Remote access from anywhere

**Your agent will:**
- ✅ Collect SPY/QQQ data every second
- ✅ Monitor positions
- ✅ Execute trades (when conditions met)
- ✅ Learn from performance
- ✅ Self-heal from errors

**You can:**
- ✅ Monitor from anywhere
- ✅ View logs remotely
- ✅ Update code anytime
- ✅ Control via SSH

---

## Quick Reference Card

```bash
# SSH to droplet
ssh trader@167.99.123.45

# View logs
tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log

# Check status
sudo systemctl status trading-agent

# Restart agent
sudo systemctl restart trading-agent

# Health check from laptop
./scripts/check_remote_health.sh 167.99.123.45
```

---

## Getting Help

**Issues?**
- Check logs: `sudo journalctl -u trading-agent -n 100`
- View errors: `tail -100 ~/Automated-Options-Trading-Agent/logs/trading_agent.log | grep -i error`

**DigitalOcean Help:**
- Tutorials: digitalocean.com/community/tutorials
- Support: Submit ticket from console

**Documentation:**
- This guide: `DIGITALOCEAN_DEPLOYMENT.md`
- Full deployment: `CLOUD_DEPLOYMENT_GUIDE.md`
- Troubleshooting: See sections above

---

**Congratulations! Your trading agent is now running 24/7 on DigitalOcean!** 🎉☁️

**Remember**: Start with paper trading, monitor for 1-2 weeks, then consider live trading!

