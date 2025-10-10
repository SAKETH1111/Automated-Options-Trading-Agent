# ✅ DigitalOcean Deployment Checklist

## Pre-Deployment

- [ ] Have Alpaca API keys ready (paper trading)
- [ ] Read `DIGITALOCEAN_DEPLOYMENT.md` 
- [ ] Ensure git repo is pushed to GitHub

---

## Step 1: Create DigitalOcean Account (5 min)

- [ ] Go to digitalocean.com
- [ ] Sign up for account
- [ ] Add payment method
- [ ] Verify email

---

## Step 2: Create Droplet (5 min)

- [ ] Click "Create" → "Droplets"
- [ ] Choose **Ubuntu 22.04 LTS**
- [ ] Select **$6/month** plan (1GB RAM)
- [ ] Choose region closest to you
- [ ] Set authentication (password or SSH key)
- [ ] Set hostname: `trading-agent-01`
- [ ] Click "Create Droplet"
- [ ] **Copy droplet IP address**: ________________

---

## Step 3: Test Connection (1 min)

- [ ] Open Terminal on your laptop
- [ ] Run: `ping YOUR_DROPLET_IP`
- [ ] Verify you get replies
- [ ] Press Ctrl+C to stop

---

## Step 4: Deploy (10 min)

### Option A: Automated (Recommended)

- [ ] Open Terminal on laptop
- [ ] Run: `cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent`
- [ ] Run: `./scripts/deploy_to_server.sh`
- [ ] Enter droplet IP when prompted
- [ ] Enter `root` for SSH user
- [ ] Wait for deployment to complete (5-10 minutes)

### Option B: Manual (If script fails)

Follow manual steps in `DIGITALOCEAN_DEPLOYMENT.md`

---

## Step 5: Configure API Keys (5 min)

- [ ] SSH to droplet: `ssh root@YOUR_DROPLET_IP`
- [ ] Switch user: `su - trader`
- [ ] Navigate: `cd Automated-Options-Trading-Agent`
- [ ] Edit config: `nano .env`
- [ ] Add your Alpaca API keys:
  ```
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
  ```
- [ ] Save file: Ctrl+X, Y, Enter

---

## Step 6: Initialize & Start (3 min)

- [ ] Activate venv: `source venv/bin/activate`
- [ ] Initialize DB: `python scripts/init_db.py`
- [ ] Run migration: `python scripts/migrate_add_tick_data.py`
- [ ] Start service: `sudo systemctl start trading-agent`
- [ ] Check status: `sudo systemctl status trading-agent`
- [ ] Verify "Active: active (running)" ✅

---

## Step 7: Verify It's Working (5 min)

- [ ] View logs: `tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log`
- [ ] Look for: "✅ Trading Agent is now LIVE"
- [ ] Look for: "RealTimeDataCollector initialized"
- [ ] Press Ctrl+C to exit log view
- [ ] Run health check: `python scripts/system_health.py`
- [ ] Verify status is HEALTHY ✅

---

## Step 8: Setup Monitoring (5 min)

### From Your Laptop:

- [ ] Create alias in `~/.zshrc`:
  ```bash
  alias trading-ssh='ssh trader@YOUR_DROPLET_IP'
  alias trading-logs='ssh trader@YOUR_DROPLET_IP "tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log"'
  alias trading-health='cd /Users/sanju/Documents/GitHub/Automated-Options-Trading-Agent && ./scripts/check_remote_health.sh YOUR_DROPLET_IP'
  ```
- [ ] Reload shell: `source ~/.zshrc`
- [ ] Test: `trading-health`

---

## Post-Deployment (Optional but Recommended)

### Security:
- [ ] Setup SSH keys (disable password auth)
- [ ] Enable firewall
- [ ] Setup automatic security updates

### Monitoring:
- [ ] Sign up for UptimeRobot.com (free)
- [ ] Add ping monitor for your droplet IP
- [ ] Configure email alerts

### Backups:
- [ ] Setup automated daily backups
- [ ] Test backup/restore process

---

## Ongoing Maintenance

### Daily:
- [ ] Check health: `trading-health`
- [ ] Review logs for errors

### Weekly:
- [ ] Check data collection stats
- [ ] Review performance metrics
- [ ] Update code if needed

### Monthly:
- [ ] Download backups
- [ ] Review costs
- [ ] Optimize if needed

---

## Quick Reference

**Your Droplet IP**: ________________

**SSH Command**: `ssh trader@YOUR_DROPLET_IP`

**View Logs**: `trading-logs` or `tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log`

**Check Health**: `trading-health` or `python scripts/system_health.py`

**Restart Agent**: `sudo systemctl restart trading-agent`

---

## Troubleshooting

### Agent not starting?
- [ ] Check logs: `sudo journalctl -u trading-agent -n 100`
- [ ] Verify .env has API keys
- [ ] Reinitialize database

### Can't SSH?
- [ ] Verify droplet is running (DigitalOcean console)
- [ ] Check IP address
- [ ] Try console access from DigitalOcean

### Out of memory?
- [ ] Add swap space (see `DIGITALOCEAN_DEPLOYMENT.md`)
- [ ] Reduce collection frequency
- [ ] Upgrade to $12/month droplet

---

## Success Criteria

✅ **Your deployment is successful if**:

- [ ] Agent service is running
- [ ] Logs show "Trading Agent is now LIVE"
- [ ] Data collection is active
- [ ] Health check shows HEALTHY
- [ ] No critical errors in logs
- [ ] You can SSH from your laptop
- [ ] You can view logs remotely

---

## Next Steps After Successful Deployment

1. **Monitor for 24 hours**
   - Check logs periodically
   - Verify data collection
   - Look for any errors

2. **Let it run for 1 week**
   - Collect data during market hours
   - Monitor performance
   - Review collected tick data

3. **Analyze performance**
   - Review trading decisions
   - Check error rates
   - Optimize configuration

4. **Consider live trading** (only after 2+ weeks of successful paper trading)
   - Change to live API
   - Start with small positions
   - Monitor closely

---

## Cost Tracking

**Monthly Cost**: $6

**Annual Cost**: $72

**Alternative**: Upgrade to $12/month for 2GB RAM if needed

---

## Important Reminders

⚠️ **Always use paper trading first** - Test for at least 2 weeks

⚠️ **Monitor regularly** - Check logs and health daily initially

⚠️ **Backup your data** - Setup automated backups

⚠️ **Keep API keys secure** - Never commit .env to git

⚠️ **Start small** - When moving to live, use small positions first

---

## Support Resources

- **Full Guide**: `DIGITALOCEAN_DEPLOYMENT.md`
- **Troubleshooting**: See guide above
- **DigitalOcean Docs**: digitalocean.com/docs
- **Community**: DigitalOcean community forums

---

**Good luck with your deployment!** 🚀

Remember: Start with paper trading, monitor closely, and scale gradually!

