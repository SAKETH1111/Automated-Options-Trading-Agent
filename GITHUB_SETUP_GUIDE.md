# 🔗 GitHub + DigitalOcean CI/CD Setup Guide

This guide will help you set up automated deployment from GitHub to your DigitalOcean droplet.

## 📋 Prerequisites

✅ DigitalOcean droplet running (IP: 45.55.150.19)  
✅ GitHub repository: `SAKETH1111/Automated-Options-Trading-Agent`  
✅ SSH access to droplet working  
✅ Trading agent already deployed and running  

## 🔑 Step 1: Configure GitHub Secrets

You need to add these secrets to your GitHub repository:

### **Go to GitHub Repository Settings:**
1. Go to: https://github.com/SAKETH1111/Automated-Options-Trading-Agent
2. Click **Settings** (top right)
3. Click **Secrets and variables** → **Actions**
4. Click **New repository secret**

### **Add These 3 Secrets:**

#### **Secret 1: DROPLET_IP**
- **Name**: `DROPLET_IP`
- **Value**: `45.55.150.19`

#### **Secret 2: DROPLET_USER**
- **Name**: `DROPLET_USER`
- **Value**: `root`

#### **Secret 3: DROPLET_SSH_KEY**
- **Name**: `DROPLET_SSH_KEY`
- **Value**: (Your private SSH key content)

To get your private SSH key:
```bash
cat ~/.ssh/id_rsa
```

Copy the entire content (including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`)

## 🚀 Step 2: Test the Setup

### **Manual Test:**
1. Go to your GitHub repository
2. Click **Actions** tab
3. Click **Deploy to DigitalOcean** workflow
4. Click **Run workflow** → **Run workflow**

### **Automatic Test:**
Make a small change and push to trigger deployment:
```bash
# Make a small change
echo "# Updated $(date)" >> README.md

# Commit and push
git add README.md
git commit -m "Test automated deployment"
git push origin main
```

## 📊 What the CI/CD Does

### **On Every Push to Main:**
1. ✅ **Stops** the trading agent service
2. ✅ **Pulls** latest code from GitHub
3. ✅ **Updates** Python dependencies
4. ✅ **Updates** database schema if needed
5. ✅ **Restarts** the trading agent service
6. ✅ **Performs** health check
7. ✅ **Reports** deployment status

### **Daily at 6 AM UTC:**
- Automatically redeploys to ensure everything is up-to-date

## 🔍 Monitoring Deployments

### **Check Deployment Status:**
1. Go to: https://github.com/SAKETH1111/Automated-Options-Trading-Agent/actions
2. Click on any workflow run to see detailed logs

### **Deployment Logs Show:**
- ✅ Service status
- ✅ Database connectivity
- ✅ Data collection status
- ✅ Memory usage
- ✅ Error messages (if any)

## 🛠️ Manual Deployment Commands

If you need to deploy manually:

```bash
# SSH into droplet
ssh root@45.55.150.19

# Navigate to app
cd /opt/trading-agent

# Run deployment script
./scripts/github_deploy.sh
```

## 🔧 Troubleshooting

### **If GitHub Actions Fail:**

1. **Check SSH Key:**
   ```bash
   ssh -T git@github.com  # Should work without password
   ```

2. **Check Droplet Access:**
   ```bash
   ssh root@45.55.150.19  # Should work without password
   ```

3. **Check Service Status:**
   ```bash
   ssh root@45.55.150.19 "systemctl status trading-agent"
   ```

### **Common Issues:**

| Issue | Solution |
|-------|----------|
| SSH key not working | Regenerate SSH key and update GitHub secret |
| Service won't start | Check logs: `journalctl -u trading-agent -f` |
| Database connection failed | Restart PostgreSQL: `systemctl restart postgresql` |
| Git pull fails | Check network connectivity |

## 🎯 Benefits of This Setup

✅ **Automated Deployments** - No manual work needed  
✅ **Version Control** - All changes tracked in GitHub  
✅ **Rollback Capability** - Easy to revert changes  
✅ **Health Checks** - Automatic verification after deployment  
✅ **Daily Updates** - Keeps system fresh  
✅ **Team Collaboration** - Multiple people can contribute  
✅ **Deployment History** - Full audit trail in GitHub  

## 📱 Quick Commands

```bash
# Check current deployment
./monitor_agent.sh

# Force deployment from GitHub
git commit --allow-empty -m "Force deployment"
git push origin main

# Check GitHub Actions status
open https://github.com/SAKETH1111/Automated-Options-Trading-Agent/actions
```

## 🎉 You're All Set!

Once you've added the GitHub secrets, your setup will be:

1. **Fully Automated** - Push to GitHub → Auto-deploy to droplet
2. **Professional** - Industry-standard CI/CD pipeline
3. **Reliable** - Health checks and error handling
4. **Scalable** - Easy to add more features and environments

**Your trading agent will now automatically update whenever you push changes to GitHub!** 🚀
