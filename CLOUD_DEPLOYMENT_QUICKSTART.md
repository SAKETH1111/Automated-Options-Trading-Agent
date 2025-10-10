# ☁️ Cloud Deployment Quick Start - 3 Easy Options

## Your Goal

Run your trading agent **24/7 without your laptop** being on.

## 🚀 Option 1: DigitalOcean (Easiest - 15 minutes)

**Cost**: $6/month | **Difficulty**: ⭐☆☆

### Step 1: Create Server (3 minutes)

1. Go to [digitalocean.com](https://digitalocean.com) → Sign up
2. Click **"Create"** → **"Droplets"**
3. Choose:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic - $6/month (1GB RAM, 1 vCPU)
   - **Region**: New York or San Francisco
   - **Authentication**: Password (for now)
4. Click **"Create Droplet"**
5. Note your **IP address** (e.g., 165.232.123.45)

### Step 2: Use Automated Deployment (2 minutes)

```bash
# From your laptop, run:
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent
./scripts/deploy_to_server.sh

# When prompted:
Enter server IP address: <your-droplet-ip>
Enter SSH user: root
```

The script will automatically:
- ✅ Install all dependencies
- ✅ Setup database
- ✅ Deploy your code
- ✅ Create background service

### Step 3: Configure & Start (5 minutes)

```bash
# SSH to your server
ssh root@YOUR_DROPLET_IP

# Switch to trader user
su - trader
cd Automated-Options-Trading-Agent

# Add your API keys
nano .env
```

Paste your keys:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Save and exit (Ctrl+X, Y, Enter)

```bash
# Start the agent
sudo systemctl start trading-agent

# Check it's running
sudo systemctl status trading-agent

# View logs
tail -f logs/trading_agent.log
```

**Done! 🎉** Your agent is now running 24/7!

### Step 4: Monitor (Ongoing)

```bash
# Check health
python scripts/system_health.py

# View logs
tail -f logs/trading_agent.log

# Restart if needed
sudo systemctl restart trading-agent
```

---

## 🐳 Option 2: Docker (Any Platform - 20 minutes)

**Cost**: Varies | **Difficulty**: ⭐⭐☆

### Why Docker?
- Works on any cloud platform
- Isolated environment
- Easy updates

### Step 1: Push to GitHub (if not done)

```bash
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent
git add .
git commit -m "Add cloud deployment"
git push origin main
```

### Step 2: Deploy to Any Server

**On your cloud server**:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone your repo
git clone https://github.com/YOUR_USERNAME/Automated-Options-Trading-Agent.git
cd Automated-Options-Trading-Agent

# Create .env with your API keys
nano .env
```

### Step 3: Start with Docker Compose

```bash
# Start everything
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop
docker-compose down
```

**Done! 🎉** Your agent runs in isolated containers!

---

## 🥧 Option 3: Raspberry Pi at Home (1 hour)

**Cost**: $80 one-time | **Difficulty**: ⭐⭐☆

### Why Raspberry Pi?
- One-time cost ($80)
- Full control
- Learn about servers
- Fun project!

### What You Need
- Raspberry Pi 4 (4GB RAM): $55
- 32GB MicroSD card: $10
- Power supply: $8
- Case: $5
- Ethernet cable (recommended)

### Setup Steps

**1. Flash OS (10 minutes)**
- Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
- Insert SD card
- Choose "Raspberry Pi OS (64-bit)"
- Click settings icon ⚙️:
  - Enable SSH
  - Set username: trader
  - Set password: your_password
  - Configure WiFi
- Flash!

**2. Boot Raspberry Pi (5 minutes)**
- Insert SD card
- Connect power
- Wait 2 minutes for boot

**3. Find IP Address**

On your laptop:
```bash
# Option A: If on same network
ping raspberrypi.local

# Option B: Check your router
# Look for "raspberrypi" in connected devices
```

**4. Deploy**

```bash
# From your laptop
cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent
./scripts/deploy_to_server.sh

# Enter Raspberry Pi IP when prompted
Enter server IP address: 192.168.1.xyz
Enter SSH user: trader
```

**5. Configure & Start**

Same as Option 1, Step 3!

**Done! 🎉** Your Pi runs the agent 24/7 at home!

### Keeping Pi Happy
- Use a UPS (battery backup) - $30
- Monitor temperature: `vcgencmd measure_temp`
- Keep it ventilated
- Use ethernet (more stable than WiFi)

---

## Quick Comparison

| Option | Cost/Month | Setup Time | Difficulty | Best For |
|--------|------------|------------|------------|----------|
| **DigitalOcean** | $6 | 15 min | ⭐☆☆ | Beginners |
| **Docker** | $6-40 | 20 min | ⭐⭐☆ | Portability |
| **Raspberry Pi** | $2 | 1 hour | ⭐⭐☆ | Learning |
| **AWS EC2** | $10 | 30 min | ⭐⭐☆ | AWS users |

**Recommendation**: Start with **DigitalOcean** - easiest and cheap!

---

## Monitoring Your Deployed Agent

### Check Status
```bash
# SSH to your server
ssh trader@YOUR_SERVER_IP

# Check if running
sudo systemctl status trading-agent

# View real-time logs
tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log

# Health check
cd ~/Automated-Options-Trading-Agent
python scripts/system_health.py
```

### Common Commands

```bash
# Start agent
sudo systemctl start trading-agent

# Stop agent
sudo systemctl stop trading-agent

# Restart agent
sudo systemctl restart trading-agent

# View logs (last 50 lines)
sudo journalctl -u trading-agent -n 50

# View logs (real-time)
sudo journalctl -u trading-agent -f
```

### Update Code

```bash
# SSH to server
ssh trader@YOUR_SERVER_IP

# Pull latest changes
cd ~/Automated-Options-Trading-Agent
git pull origin main

# Restart
sudo systemctl restart trading-agent
```

---

## Troubleshooting

### Agent Won't Start

```bash
# Check logs for errors
sudo journalctl -u trading-agent -n 100

# Common issues:
# 1. Missing API keys in .env
# 2. Database not initialized
# 3. Python dependencies missing

# Fix: Run setup again
cd ~/Automated-Options-Trading-Agent
source venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py
```

### Can't Connect to Server

```bash
# Test connection
ping YOUR_SERVER_IP

# If timeout:
# - Check server is running
# - Check firewall allows SSH (port 22)
# - Check IP address is correct
```

### Out of Memory

```bash
# Check memory
free -h

# If low, add swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## Security Checklist

### Essential Security
- [ ] Change default passwords
- [ ] Setup SSH keys (disable password auth)
- [ ] Enable firewall
- [ ] Keep system updated
- [ ] Use strong database password

### Setup SSH Keys (Recommended)

**On your laptop**:
```bash
# Generate key if you don't have one
ssh-keygen -t ed25519

# Copy to server
ssh-copy-id trader@YOUR_SERVER_IP

# Test
ssh trader@YOUR_SERVER_IP
```

### Enable Firewall

**On server**:
```bash
sudo ufw allow 22/tcp  # SSH
sudo ufw enable
```

---

## Cost Breakdown

### DigitalOcean
- **Basic Droplet**: $6/month
- **Regular Droplet**: $12/month (2GB RAM)
- **Premium**: $24/month (4GB RAM)

**Total**: $6-24/month

### AWS EC2
- **t3.micro**: $8/month (1GB RAM)
- **t3.small**: $17/month (2GB RAM)
- **t3.medium**: $34/month (4GB RAM)

**Total**: $8-34/month

### Raspberry Pi
- **Hardware**: $80 one-time
- **Electricity**: ~$2/month
- **Internet**: Included in home internet

**Total**: $80 + $2/month

### Recommended: DigitalOcean $6/month
- Cheapest ongoing cost
- Very reliable
- Easy to setup
- Can upgrade anytime

---

## Next Steps After Deployment

### 1. Setup Alerts (Recommended)

Use [UptimeRobot](https://uptimerobot.com) (Free):
- Monitor your server every 5 minutes
- Email alerts if down
- SMS alerts (paid)

### 2. Setup Backups (Important)

```bash
# Create backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
cd ~/Automated-Options-Trading-Agent
tar -czf ~/backup_$(date +%Y%m%d).tar.gz data/ logs/ .env
```

```bash
chmod +x ~/backup.sh

# Run daily at 2 AM
crontab -e
# Add: 0 2 * * * /home/trader/backup.sh
```

### 3. Monitor Performance

```bash
# Install htop
sudo apt install htop

# Monitor resources
htop

# Check disk space
df -h

# Check memory
free -h
```

---

## Summary

**To run without your laptop**:

1. ✅ **Choose**: DigitalOcean $6/month (easiest)
2. ✅ **Deploy**: Use `./scripts/deploy_to_server.sh`
3. ✅ **Configure**: Add API keys in `.env`
4. ✅ **Start**: `sudo systemctl start trading-agent`
5. ✅ **Monitor**: `tail -f logs/trading_agent.log`

**Your agent now runs 24/7 independently!** 🚀

---

## Getting Help

**Deployment Issues**:
- Check logs: `sudo journalctl -u trading-agent -f`
- Test manually: `python main.py`

**DigitalOcean Help**:
- Extensive tutorials
- Community forum
- Support tickets

**General Help**:
- Full guide: `CLOUD_DEPLOYMENT_GUIDE.md`
- This quickstart: `CLOUD_DEPLOYMENT_QUICKSTART.md`

**Start with Option 1 (DigitalOcean) - it's the easiest!** ☁️

