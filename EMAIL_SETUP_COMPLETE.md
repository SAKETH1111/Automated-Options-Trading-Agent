# 📧 Email Alerts Setup - Complete Guide

## ✅ **Email Configuration: DONE**

Your email alerts are configured on the droplet!

**Configuration:**
- ✅ Sender Email: saketh1111@gmail.com
- ✅ Recipient Email: saketh.kkp40@gmail.com
- ✅ App Password: Configured
- ✅ Code: Updated and deployed

---

## ⚠️ **Issue: DigitalOcean Blocks SMTP Ports**

DigitalOcean blocks outgoing SMTP ports (25, 587) to prevent spam.

**This is a common issue with cloud providers.**

---

## 🔧 **Solutions (Choose One):**

### **Solution 1: Use SendGrid (Recommended - FREE)**

SendGrid provides 100 emails/day for free and works with DigitalOcean.

#### **Setup Steps:**
1. **Sign up**: https://signup.sendgrid.com/
2. **Get API Key**: 
   - Go to Settings → API Keys
   - Create API Key
   - Copy the key

3. **Update .env file:**
```bash
ssh root@45.55.150.19 "nano /opt/trading-agent/.env"

# Replace email section with:
SENDGRID_API_KEY=your_sendgrid_api_key_here
ALERT_EMAIL=saketh1111@gmail.com
ALERT_RECIPIENT_EMAIL=saketh.kkp40@gmail.com
```

4. **Install SendGrid:**
```bash
ssh root@45.55.150.19 "cd /opt/trading-agent && source venv/bin/activate && pip install sendgrid"
```

5. **Update code** (I can do this for you)

### **Solution 2: Use Mailgun (Alternative - FREE tier)**

Similar to SendGrid, Mailgun also works.

- **Free tier**: 5,000 emails/month
- **Setup**: https://www.mailgun.com/
- **Works with**: DigitalOcean

### **Solution 3: Use Port 465 (SSL)**

Try Gmail's SSL port instead of TLS:

```bash
# Update email_alerts.py to use port 465 with SSL
# I can make this change
```

### **Solution 4: Use External SMTP Relay**

Use a service like:
- **Mailjet** (FREE 200 emails/day)
- **Elastic Email** (FREE 100 emails/day)
- **SMTP2GO** (FREE 1,000 emails/month)

---

## 🎯 **My Recommendation: SendGrid**

**Why SendGrid:**
- ✅ FREE (100 emails/day)
- ✅ Works with DigitalOcean
- ✅ Reliable delivery
- ✅ Easy to set up
- ✅ Good documentation

**Setup time**: 10 minutes

---

## 📋 **Quick Setup: SendGrid**

### **Step 1: Sign Up**
Go to: https://signup.sendgrid.com/

### **Step 2: Verify Email**
Check your email and verify your account

### **Step 3: Create API Key**
1. Go to Settings → API Keys
2. Click "Create API Key"
3. Name it: "Trading Agent"
4. Select "Full Access"
5. **Copy the API key** (you'll only see it once!)

### **Step 4: Configure**
```bash
ssh root@45.55.150.19 "nano /opt/trading-agent/.env"

# Add at the end:
SENDGRID_API_KEY=SG.your_api_key_here
```

### **Step 5: I'll Update the Code**
Just tell me when you have the SendGrid API key and I'll update the email alerts to use it!

---

## 🎯 **Alternative: Telegram Bot (Even Better!)**

**Instead of email, use Telegram:**

**Advantages:**
- ✅ FREE forever
- ✅ Instant notifications
- ✅ No port blocking issues
- ✅ Two-way communication
- ✅ Can control your agent from phone
- ✅ Better than email

**Setup time**: 5 minutes

**Commands you'll have:**
```
/status - Get current status
/positions - View open positions
/pnl - Check P&L
/stop - Stop trading
/start - Start trading
/risk - View risk metrics
```

---

## 💡 **My Recommendation:**

### **Best Option: Telegram Bot**
- Easier to set up
- No port blocking issues
- More features
- Better UX
- FREE

### **Good Option: SendGrid**
- If you prefer email
- Professional
- Reliable
- FREE tier sufficient

---

## 🎯 **What Would You Like?**

**Option 1: Set up SendGrid for email** (10 minutes)
- I'll walk you through getting API key
- I'll update the code
- You'll get email alerts

**Option 2: Build Telegram bot** (I can do this now - 30 minutes)
- Even better than email
- Instant notifications
- Control from phone
- No port issues

**Option 3: Both** (Best of both worlds)
- Telegram for instant alerts
- Email for daily summaries

---

## 📊 **Current Status:**

✅ **Email code**: Configured and ready  
✅ **Credentials**: Saved on droplet  
⚠️ **Blocked**: DigitalOcean blocks SMTP ports  
✅ **Solution**: Use SendGrid or Telegram  

---

**What would you like to do?**
1. Set up SendGrid (I'll guide you)
2. Build Telegram bot (I'll do it now)
3. Both SendGrid + Telegram
4. Skip alerts for now

Let me know! 🚀

