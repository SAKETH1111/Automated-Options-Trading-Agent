# ⚡ Quick Start Guide

Get up and running with the Automated Options Trading Agent in 5 minutes!

## Prerequisites

- Python 3.9+
- Alpaca Paper Trading Account
- 5 minutes of your time ⏱️

## Installation

### 1. Clone & Install

```bash
# Clone repository
git clone https://github.com/yourusername/Automated-Options-Trading-Agent.git
cd Automated-Options-Trading-Agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` file:

```bash
# Get your keys from: https://alpaca.markets
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Initialize & Verify

```bash
# Initialize database
python scripts/init_db.py

# Verify setup
python scripts/check_setup.py
```

You should see:
```
✅ All checks passed! Ready to start trading.
```

### 4. Start Trading

```bash
python main.py
```

That's it! 🎉 The agent is now running.

## What Happens Next?

The agent will:
1. ✅ Wait for market open (9:30 AM ET)
2. 🔍 Scan watchlist for opportunities
3. 📊 Generate trading signals
4. ✔️ Check risk constraints
5. 💰 Execute trades automatically
6. 📈 Monitor positions continuously
7. 🎯 Exit at take-profit or stop-loss
8. 📚 Learn from each trade

## Monitoring

### View Logs

```bash
# Follow live logs
tail -f logs/trading_agent.log

# View trade journal
tail -f logs/trade_journal.jsonl
```

### Check Status

```python
from src.orchestrator import TradingOrchestrator

orchestrator = TradingOrchestrator()
status = orchestrator.get_status()

print(f"Running: {status['is_running']}")
print(f"Positions: {status['portfolio']['total_positions']}")
print(f"P&L: ${status['portfolio']['total_unrealized_pnl']:.2f}")
```

## First Steps

### Day 1-7: Observe

- Let it run in paper trading
- Watch the logs
- See what signals it generates
- Understand the trade flow

### Week 2-4: Learn

- Review closed trades
- Check performance metrics
- Read the learning insights
- Adjust parameters if needed

### Month 2-3: Optimize

- Enable additional strategies
- Expand watchlist
- Fine-tune risk parameters
- Track improvement

## Configuration Presets

### Conservative (Recommended for Beginners)

```yaml
# config/config.yaml
trading:
  risk:
    max_daily_loss_pct: 2.0
    max_trades_per_day: 2
    max_position_size_pct: 5.0

strategies:
  bull_put_spread:
    enabled: true
    short_delta_range: [-0.25, -0.20]
    min_credit: 0.40
```

### Moderate

```yaml
trading:
  risk:
    max_daily_loss_pct: 3.0
    max_trades_per_day: 5
    max_position_size_pct: 10.0

strategies:
  bull_put_spread:
    enabled: true
  cash_secured_put:
    enabled: true
```

### Aggressive

```yaml
trading:
  risk:
    max_daily_loss_pct: 5.0
    max_trades_per_day: 10
    max_position_size_pct: 20.0

strategies:
  bull_put_spread:
    enabled: true
  cash_secured_put:
    enabled: true
  iron_condor:
    enabled: true
```

## Common Commands

```bash
# Start agent
python main.py

# Initialize database
python scripts/init_db.py

# Check setup
python scripts/check_setup.py

# Run tests
pytest tests/ -v

# View logs
tail -f logs/trading_agent.log

# Stop agent
# Press Ctrl+C
```

## Getting Help

### Documentation

- 📖 [Setup Guide](docs/SETUP_GUIDE.md) - Detailed setup
- 📊 [Strategy Guide](docs/STRATEGY_GUIDE.md) - Strategy details
- 📚 [Full README](README.md) - Complete documentation

### Troubleshooting

**"Can't connect to Alpaca"**
- Check API keys in `.env`
- Verify using paper trading URL

**"No signals generated"**
- Market might be closed
- Check if symbols in watchlist have options
- Try with SPY or QQQ

**"Database error"**
- Run `python scripts/init_db.py`
- Check DATABASE_URL in `.env`

### Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/Automated-Options-Trading-Agent/issues)
- Documentation: `docs/` folder
- Examples: `scripts/` folder

## Next Steps

### Learn the System

1. Read [Strategy Guide](docs/STRATEGY_GUIDE.md)
2. Understand risk management
3. Review sample trades
4. Experiment with parameters

### Customize

1. Edit watchlist in `config/config.yaml`
2. Adjust risk parameters
3. Enable/disable strategies
4. Set up alerts

### Monitor & Improve

1. Check daily performance
2. Review learning insights
3. Optimize parameters
4. Track metrics

## Important Reminders

⚠️ **Start with Paper Trading**
- Never use real money initially
- Test for 3+ months
- Understand the system completely

⚠️ **Risk Management**
- Never risk more than 1-2% per trade
- Set daily loss limits
- Use stop losses

⚠️ **Continuous Learning**
- Review every trade
- Understand losses
- Adjust and improve

---

**Happy Trading! 🚀📈**

*Remember: This is your learning journey. Start small, be patient, and continuously improve!*





