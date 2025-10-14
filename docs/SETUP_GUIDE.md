# 📖 Setup Guide

Complete guide to setting up the Automated Options Trading Agent.

## Prerequisites

### System Requirements
- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- 10GB disk space
- Stable internet connection

### Required Accounts
1. **Alpaca Account** (required)
   - Sign up at [alpaca.markets](https://alpaca.markets)
   - Start with paper trading account
   - Generate API keys from dashboard

2. **PostgreSQL** (optional)
   - Can use SQLite (default)
   - PostgreSQL recommended for production

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Automated-Options-Trading-Agent.git
cd Automated-Options-Trading-Agent
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `.env` file:

```bash
# Copy example (if using .env.example)
cp .env.example .env

# Edit with your settings
nano .env
```

Required environment variables:

```bash
# Alpaca API Credentials (REQUIRED)
ALPACA_API_KEY=PKxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database (optional, defaults to SQLite)
DATABASE_URL=sqlite:///./trading_agent.db

# Trading Mode
TRADING_MODE=paper

# Risk Parameters
MAX_DAILY_LOSS_PCT=5.0
MAX_POSITION_SIZE_PCT=20.0
MAX_TRADES_PER_DAY=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_agent.log

# Alerts (optional)
ALERT_EMAIL=your-email@example.com
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Learning
ENABLE_LEARNING=true
LEARNING_UPDATE_FREQUENCY=daily
```

### 5. Configure Strategies

Edit `config/config.yaml`:

```yaml
# Start with conservative settings
strategies:
  bull_put_spread:
    enabled: true
    dte_range: [30, 45]
    short_delta_range: [-0.25, -0.20]  # More conservative
    width_range: [5, 10]
    min_credit: 0.40
    max_risk_reward: 3.0
    
  cash_secured_put:
    enabled: false  # Start with just one strategy
    
  iron_condor:
    enabled: false  # Enable later

trading:
  risk:
    max_daily_loss_pct: 3.0  # Start conservative
    max_position_size_pct: 10.0
    max_trades_per_day: 3
```

### 6. Initialize Database

```bash
python scripts/init_db.py
```

You should see:
```
✅ Database initialized successfully
```

### 7. Verify Setup

```bash
python scripts/check_setup.py
```

Expected output:
```
✅ Configuration OK
✅ Connected to Alpaca
   Account Equity: $100,000.00
   Buying Power: $100,000.00
✅ Database connection OK
✅ All checks passed! Ready to start trading.
```

### 8. Test Run (Dry Run)

Before starting the agent, test signal generation:

```python
from src.signals.generator import SignalGenerator

generator = SignalGenerator()
signals = generator.scan_for_signals(symbols=["SPY"])

for signal in signals:
    print(f"Signal: {signal['strategy_name']} on {signal['symbol']}")
    print(f"Quality: {signal['signal_quality']}/100")
```

### 9. Start the Agent

```bash
python main.py
```

Monitor logs:
```bash
tail -f logs/trading_agent.log
```

## Configuration Best Practices

### For Beginners

1. **Start Small**
   - Use paper trading only
   - Enable only one strategy (Bull Put Spread)
   - Limit to 1-2 trades per day
   - Watch for 2-4 weeks

2. **Conservative Settings**
   ```yaml
   risk:
     max_daily_loss_pct: 2.0
     max_position_size_pct: 5.0
     max_trades_per_day: 2
     stop_loss_pct: 50
     take_profit_pct: 50
   ```

3. **Limited Watchlist**
   - Start with liquid ETFs: SPY, QQQ
   - Avoid earnings weeks
   - Focus on high IV rank (>40)

### For Experienced Users

1. **Multiple Strategies**
   - Enable multiple strategies after testing individually
   - Diversify across different market conditions
   - Adjust parameters based on backtests

2. **Advanced Risk Management**
   ```yaml
   risk:
     max_portfolio_heat: 25.0
     use_kelly_criterion: true
     dynamic_position_sizing: true
   ```

3. **Expanded Watchlist**
   - 15-20 liquid symbols
   - Mix of indices, large caps
   - Sector diversification

## Docker Deployment

### Using Docker Compose

1. **Configure Environment**

Create `.env` file with your settings.

2. **Build and Start**

```bash
docker-compose up -d
```

3. **View Logs**

```bash
docker-compose logs -f trading-agent
```

4. **Access Database**

```bash
docker-compose exec postgres psql -U trading_user -d options_trading
```

5. **Stop Services**

```bash
docker-compose down
```

### Production Docker Setup

For production, use:

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  trading-agent:
    build: .
    restart: always
    env_file:
      - .env.production
    volumes:
      - /path/to/logs:/app/logs
      - /path/to/data:/app/data
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Troubleshooting

### Common Issues

1. **"Alpaca connection error"**
   - Verify API keys in `.env`
   - Check if keys are for paper trading
   - Ensure API keys have required permissions

2. **"No options data"**
   - Alpaca may not have options data for some symbols
   - Try SPY or QQQ first
   - Check if market is open

3. **"Database connection failed"**
   - Verify DATABASE_URL format
   - Check PostgreSQL is running
   - Try SQLite first: `DATABASE_URL=sqlite:///./trading_agent.db`

4. **"Daily trade limit reached"**
   - Adjust `MAX_TRADES_PER_DAY` in config
   - Wait for next trading day
   - Or reset manually in database

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG python main.py
```

### Getting Help

1. Check logs: `logs/trading_agent.log`
2. Run diagnostics: `python scripts/check_setup.py`
3. Test individual components
4. Open GitHub issue with logs

## Monitoring

### Real-time Monitoring

```bash
# Main logs
tail -f logs/trading_agent.log

# Trade journal
tail -f logs/trade_journal.jsonl

# System metrics
watch -n 5 'python -c "from src.orchestrator import TradingOrchestrator; o = TradingOrchestrator(); print(o.get_status())"'
```

### Daily Checklist

- [ ] Check account balance
- [ ] Review open positions
- [ ] Monitor daily P&L
- [ ] Check for alerts/errors
- [ ] Verify scheduled tasks ran

### Weekly Review

- [ ] Analyze performance metrics
- [ ] Review learning insights
- [ ] Adjust strategy parameters if needed
- [ ] Check system health
- [ ] Update watchlist

## Next Steps

1. **Paper Trading Phase**
   - Run for 3 months minimum
   - Track all metrics
   - Aim for positive expectancy
   - Build confidence in system

2. **Optimization Phase**
   - Use learning system insights
   - A/B test parameter changes
   - Optimize for your risk tolerance
   - Document all changes

3. **Live Trading Preparation**
   - Review all trades manually
   - Understand every loss
   - Verify risk management working
   - Start with minimal capital

4. **Live Trading**
   - Switch to live API keys
   - Update `TRADING_MODE=live`
   - Start with 1-2 positions
   - Monitor closely for first month

## Security Best Practices

1. **API Keys**
   - Never commit `.env` to git
   - Use paper trading keys initially
   - Rotate keys periodically
   - Limit API key permissions

2. **Access Control**
   - Secure the server
   - Use firewall rules
   - Enable authentication
   - Monitor access logs

3. **Data Backup**
   - Backup database regularly
   - Store logs securely
   - Keep configuration versioned
   - Test restore procedures

## Support Resources

- **Documentation**: `docs/`
- **Examples**: `scripts/`
- **Tests**: `tests/`
- **GitHub Issues**: Report bugs and request features
- **Community**: Join discussions

---

**Ready to trade? Start with paper trading and work your way up! 🚀**









