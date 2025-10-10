# 🚀 Automated Options Trading Agent

An intelligent, self-learning options trading system that autonomously analyzes markets, executes trades, and continuously improves through post-trade analysis.

## ✨ Features

### 🎯 Core Capabilities
- **Automated Market Scanning**: Continuously scans watchlist for high-probability setups
- **Multi-Strategy Support**: Bull Put Spreads, Cash Secured Puts, Iron Condors
- **Risk Management**: Position sizing, daily loss limits, portfolio heat monitoring
- **Intelligent Execution**: Limit orders with slippage control and multi-leg spread execution
- **Position Monitoring**: Real-time monitoring with automatic exits (take-profit, stop-loss)

### 🧠 Learning System
- **Trade Analysis**: Automatic categorization of winning and losing trades
- **Error Taxonomy**: Tracks issues across 6 categories (entry quality, liquidity, volatility, risk, timing, greeks)
- **Parameter Optimization**: Suggests strategy adjustments based on historical performance
- **Performance Metrics**: Sharpe ratio, Sortino ratio, profit factor, win rate tracking

### 📊 Data & Analytics
- **Greeks Calculation**: Delta, gamma, theta, vega, rho for all positions
- **IV Analysis**: Implied volatility calculation and IV rank tracking
- **Market Data**: Real-time quotes, options chains, historical data via Alpaca
- **Trade Journal**: Complete audit trail of every decision and execution

### 🔔 Monitoring & Alerts
- **Logging**: Structured JSON logging with rotation
- **Alerts**: Configurable alerts via console, email, webhook (Slack/Discord)
- **Dashboard**: Real-time portfolio status and risk metrics

## 📋 Prerequisites

- Python 3.9+
- Alpaca API account (paper or live)
- PostgreSQL (optional, defaults to SQLite)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Automated-Options-Trading-Agent.git
cd Automated-Options-Trading-Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```bash
# Alpaca API Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Database (optional, defaults to SQLite)
DATABASE_URL=postgresql://user:password@localhost:5432/options_trading

# Trading Configuration
TRADING_MODE=paper
MAX_DAILY_LOSS_PCT=5.0
MAX_TRADES_PER_DAY=10

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_agent.log

# Learning
ENABLE_LEARNING=true
```

### 4. Initialize Database

```bash
python scripts/init_db.py
```

### 5. Verify Setup

```bash
python scripts/check_setup.py
```

### 6. Start Trading Agent

```bash
python main.py
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-agent

# Stop services
docker-compose down
```

## ⚙️ Configuration

### Strategy Configuration

Edit `config/config.yaml` to customize strategies:

```yaml
strategies:
  bull_put_spread:
    enabled: true
    dte_range: [25, 45]
    short_delta_range: [-0.30, -0.20]
    width_range: [5, 10]
    min_credit: 0.30
    max_risk_reward: 4.0
```

### Risk Management

```yaml
trading:
  risk:
    max_daily_loss_pct: 5.0
    max_position_size_pct: 20.0
    max_trades_per_day: 10
    max_positions_per_symbol: 2
    stop_loss_pct: 50
    take_profit_pct: 50
```

### Market Scanning

```yaml
scanning:
  min_stock_price: 20
  max_stock_price: 500
  min_market_cap: 5000000000
  min_avg_volume: 1000000
  min_open_interest: 100
  min_volume: 50
  max_bid_ask_spread_pct: 10.0
  min_iv_rank: 25
  
  watchlist:
    - SPY
    - QQQ
    - AAPL
    - MSFT
    # ... add more symbols
```

## 📊 Usage Examples

### Manual Signal Generation

```python
from src.signals.generator import SignalGenerator

generator = SignalGenerator()
signals = generator.scan_for_signals(symbols=["SPY", "QQQ"])

for signal in signals:
    print(f"{signal['symbol']}: {signal['strategy_name']}")
    print(f"  Quality: {signal['signal_quality']}/100")
    print(f"  Max Profit: ${signal['max_profit']:.2f}")
    print(f"  Max Loss: ${signal['max_loss']:.2f}")
```

### Manual Trade Execution

```python
from src.execution.executor import TradeExecutor

executor = TradeExecutor()
trade_id = executor.execute_signal(signal)

if trade_id:
    print(f"Trade executed: {trade_id}")
```

### Portfolio Status

```python
from src.orchestrator import TradingOrchestrator

orchestrator = TradingOrchestrator()
status = orchestrator.get_status()

print(f"Open Positions: {status['portfolio']['total_positions']}")
print(f"Total P&L: ${status['portfolio']['total_unrealized_pnl']:.2f}")
```

### Learning Insights

```python
from src.learning.analyzer import TradeAnalyzer

analyzer = TradeAnalyzer()
metrics = analyzer.calculate_performance_metrics(period_days=30)

print(f"Win Rate: {metrics['win_rate']:.1f}%")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

## 🧪 Testing

Run tests:

```bash
pytest tests/ -v
```

Run specific test:

```bash
pytest tests/test_strategies.py::TestBullPutSpreadStrategy -v
```

## 📈 Backtesting

```bash
python scripts/backtest.py
```

*Note: The current backtest framework is simplified. For production backtesting, integrate with libraries like Backtrader or Zipline.*

## 🗂️ Project Structure

```
Automated-Options-Trading-Agent/
├── src/
│   ├── brokers/          # Alpaca API integration
│   ├── config/           # Configuration management
│   ├── database/         # Database models and session
│   ├── execution/        # Trade execution and monitoring
│   ├── learning/         # Learning and adaptation system
│   ├── market_data/      # Data collection and enrichment
│   ├── monitoring/       # Logging and alerts
│   ├── risk/             # Risk management
│   ├── signals/          # Signal generation engine
│   ├── strategies/       # Strategy implementations
│   └── orchestrator.py   # Main orchestrator
├── config/
│   └── config.yaml       # Strategy and risk configuration
├── scripts/              # Utility scripts
├── tests/                # Unit and integration tests
├── logs/                 # Log files
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
└── README.md
```

## 📚 Architecture

### Component Flow

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator                           │
│  ┌────────────────────────────────────────────────┐    │
│  │  Market Hours Scheduler (APScheduler)          │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
              │                    │                  │
              ▼                    ▼                  ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐
    │ Signal Generator │  │ Position Monitor│  │  Learning  │
    └─────────────────┘  └─────────────────┘  └────────────┘
              │                    │                  │
              ▼                    ▼                  ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐
    │   Strategies    │  │ Trade Executor  │  │ Analyzer   │
    └─────────────────┘  └─────────────────┘  └────────────┘
              │                    │                  │
              └────────────────────┴──────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Market Data Collector │
                    └─────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      Alpaca API         │
                    └─────────────────────────┘
```

### Trade Lifecycle

1. **Signal Generation**: Scan watchlist → Filter by criteria → Generate signals
2. **Risk Check**: Verify position limits → Check daily loss → Validate portfolio heat
3. **Execution**: Calculate position size → Place orders → Confirm fills
4. **Monitoring**: Update positions → Check exit conditions → Execute exits
5. **Analysis**: Categorize errors → Calculate metrics → Generate insights
6. **Learning**: Identify patterns → Suggest adjustments → Update parameters

## 🎯 Success Criteria (from PROJECT_END_GOAL.md)

- ✅ **3+ months paper trading** with positive expectancy before going live
- ✅ **≤ 1–2% loss per trade** maximum risk
- ✅ **≤ 5% daily drawdown cap**
- ✅ **30–50% fewer repeat errors** through learning system
- ✅ **Low maintenance** with automated weekly/monthly reviews

## 🔒 Risk Disclaimer

**IMPORTANT**: This software is for educational purposes only. Trading options involves substantial risk of loss. Past performance does not guarantee future results.

- Start with **paper trading** only
- Never risk more than you can afford to lose
- Thoroughly test and understand the system before using real money
- The authors are not responsible for any financial losses

## 🛠️ Development

### Adding a New Strategy

1. Create strategy file in `src/strategies/`
2. Inherit from `Strategy` base class
3. Implement required methods: `generate_signals`, `should_exit`, `should_roll`
4. Add strategy to `SignalGenerator`
5. Add configuration to `config/config.yaml`
6. Write tests

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Alpaca for providing excellent API and paper trading
- Options education resources: tastytrade, OptionAlpha
- Python options libraries: py_vollib, mibian

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

## 🗺️ Roadmap

### Phase 1 (Complete)
- [x] Core infrastructure
- [x] Basic strategies
- [x] Risk management
- [x] Learning system

### Phase 2 (In Progress)
- [ ] Enhanced backtesting
- [ ] More strategy types
- [ ] Web dashboard
- [ ] Advanced analytics

### Phase 3 (Planned)
- [ ] Machine learning models
- [ ] Multi-broker support
- [ ] Mobile alerts
- [ ] Community strategy sharing

---

**Built with ❤️ for algorithmic options trading**
