# 📊 Strategy Guide

Complete guide to the trading strategies implemented in the system.

## Implemented Strategies

### 1. Bull Put Spread (Vertical Credit Spread)

**Market Outlook**: Neutral to bullish

**Structure**:
- Sell OTM put (short leg)
- Buy further OTM put (long leg) for protection
- Credit spread - receive premium upfront

**Example**:
```
SPY @ $450
Sell $440 put @ $3.00 (credit)
Buy $435 put @ $2.00 (debit)
Net credit: $1.00 ($100 per contract)
Max loss: $4.00 ($400 per contract) if SPY < $435
```

**Entry Criteria**:
- DTE: 25-45 days
- Short delta: -0.20 to -0.30
- IV Rank: > 25
- Width: 5-10 points
- Minimum credit: $0.30
- Good liquidity (OI > 100, volume > 50)

**Exit Rules**:
- Take profit: 50% of max profit
- Stop loss: 100% of credit received (2x loss)
- Roll if threatened within 7 DTE

**Best For**:
- Neutral to slightly bullish markets
- High IV environments
- Liquid underlyings

**Configuration**:
```yaml
bull_put_spread:
  enabled: true
  dte_range: [25, 45]
  short_delta_range: [-0.30, -0.20]
  width_range: [5, 10]
  min_credit: 0.30
  max_risk_reward: 4.0
  take_profit_pct: 50
  stop_loss_pct: 100
```

---

### 2. Cash Secured Put

**Market Outlook**: Bullish, willing to own stock

**Structure**:
- Sell OTM put
- Hold cash to cover assignment
- Collect premium

**Example**:
```
AAPL @ $180
Sell $170 put @ $2.50
Premium collected: $250 per contract
If assigned: Buy 100 shares @ $170 (net cost: $167.50)
```

**Entry Criteria**:
- DTE: 30-45 days
- Delta: -0.20 to -0.30
- IV Rank: > 30
- Minimum premium: $0.50
- Stock you'd like to own

**Exit Rules**:
- Take profit: 50% of max profit
- Stop loss: 100% of premium
- Assignment is acceptable outcome

**Best For**:
- Building long positions
- Stocks you're bullish on long-term
- Income generation

**Configuration**:
```yaml
cash_secured_put:
  enabled: true
  dte_range: [30, 45]
  delta_range: [-0.30, -0.20]
  min_premium: 0.50
  take_profit_pct: 50
  stop_loss_pct: 100
```

---

### 3. Iron Condor

**Market Outlook**: Neutral, expect low volatility

**Structure**:
- Sell OTM put spread
- Sell OTM call spread
- Profit from time decay in range

**Example**:
```
SPY @ $450
Sell $440/$435 put spread: +$1.00
Sell $460/$465 call spread: +$1.00
Total credit: $2.00 ($200 per contract)
Max loss: $3.00 ($300) if SPY moves beyond strikes
```

**Entry Criteria**:
- DTE: 30-45 days
- Put delta: -0.15 to -0.20
- Call delta: 0.15 to 0.20
- IV Rank: > 30
- Width: 5 points each side
- Minimum total credit: $0.50

**Exit Rules**:
- Take profit: 50% of max profit
- Stop loss: 100% of credit
- Close if either side threatened

**Best For**:
- High IV environments
- Range-bound markets
- Neutral outlook

**Configuration**:
```yaml
iron_condor:
  enabled: true
  dte_range: [30, 45]
  short_put_delta_range: [-0.20, -0.15]
  short_call_delta_range: [0.15, 0.20]
  width: 5
  min_credit: 0.50
  take_profit_pct: 50
  stop_loss_pct: 100
```

---

## Strategy Selection Guide

### By Market Condition

| Condition | Best Strategy | Why |
|-----------|---------------|-----|
| Bullish | Bull Put Spread, CSP | Profit from upward movement |
| Neutral | Iron Condor | Profit from time decay |
| High IV | All strategies | Higher premiums available |
| Low IV | Wait | Premium not worth risk |
| Trending | Bull Put Spread | Directional bias |
| Range-bound | Iron Condor | Profit from staying in range |

### By Experience Level

**Beginner**: Start with Bull Put Spread
- Defined risk
- Easy to understand
- Good risk/reward

**Intermediate**: Add Cash Secured Put
- Requires more capital
- Stock assignment possible
- Long-term strategy

**Advanced**: Add Iron Condor
- More complex management
- Multiple legs
- Requires active monitoring

## Risk Management by Strategy

### Bull Put Spread
- **Max Loss**: (Width - Credit) × 100
- **Position Sizing**: 1-2% account risk per trade
- **Stop Loss**: 2× credit received
- **Max Positions**: 3-5 per symbol

### Cash Secured Put
- **Max Loss**: (Strike - Premium) × 100
- **Position Sizing**: Only if willing to own stock
- **Capital Required**: Strike price × 100
- **Max Positions**: 1-2 per symbol

### Iron Condor
- **Max Loss**: (Width - Credit) × 100
- **Position Sizing**: 1-2% account risk
- **Stop Loss**: 2× credit received
- **Max Positions**: 2-3 total (capital intensive)

## Common Adjustments

### Rolling

**When to Roll**:
- Position threatened (price near short strike)
- 7-10 DTE remaining
- IV still elevated
- Can collect additional credit

**How to Roll**:
1. Close current position
2. Open new position at same strikes, later expiration
3. Collect net credit
4. Extend time for position to work

**Example**:
```
Current: SPY 440/435 put spread, 7 DTE, SPY @ $442
Roll to: SPY 440/435 put spread, 35 DTE
Collect additional $0.50 credit
```

### Legging Out

**When**:
- One side of spread threatened
- Want to reduce risk
- Lock in partial profits

**Example**:
```
Iron Condor threatened on put side
Close put spread: -$150 loss
Keep call spread: +$100 profit
Net: -$50 vs potential -$300 max loss
```

## Advanced Techniques

### 1. Scaling In

Start with smaller position, add if profitable:
```
Week 1: 1 contract @ $1.00 credit
Week 2: 1 contract @ $1.00 credit
Week 3: 2 contracts @ $1.00 credit
Total: 4 contracts, $400 credit
```

### 2. Staggered Expirations

Multiple positions at different DTEs:
```
Position 1: 30 DTE
Position 2: 45 DTE
Position 3: 60 DTE
Benefits: Smooth P&L, consistent activity
```

### 3. Delta-Neutral Portfolio

Balance directional risk:
```
Bull Put Spread: -0.10 delta
Iron Condor: 0.00 delta
Total portfolio delta: -0.10 (nearly neutral)
```

## Strategy Performance Metrics

Track these metrics for each strategy:

### Required Metrics
- **Win Rate**: % of profitable trades
- **Profit Factor**: Total wins / Total losses
- **Average Win**: Mean profit on winning trades
- **Average Loss**: Mean loss on losing trades
- **Max Drawdown**: Largest peak-to-trough decline

### Advanced Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Expectancy**: (Win% × Avg Win) - (Loss% × Avg Loss)
- **Recovery Factor**: Net Profit / Max Drawdown

### Target Benchmarks

| Metric | Target | Excellent |
|--------|--------|-----------|
| Win Rate | > 60% | > 70% |
| Profit Factor | > 1.5 | > 2.0 |
| Sharpe Ratio | > 1.0 | > 1.5 |
| Max Drawdown | < 15% | < 10% |

## Backtesting Strategies

Before enabling a strategy live:

1. **Historical Analysis**
   - Test on 1+ year of data
   - Include different market conditions
   - Account for commissions/slippage

2. **Walk-Forward Testing**
   - Optimize on training data
   - Test on out-of-sample data
   - Verify parameters are robust

3. **Paper Trading**
   - Run for 3+ months
   - Track all trades
   - Verify system works as expected

## Strategy Optimization

### Parameter Tuning

Test ranges:
```python
# Bull Put Spread optimization
dte_range: [20-50]
delta_range: [-0.35 to -0.15]
width_range: [5-15]
min_credit: [0.20-0.50]
```

### Learning from Trades

Analyze:
- Winning vs losing characteristics
- Entry timing (IV rank correlation)
- Exit timing (DTE at close)
- Greeks exposure at entry
- Market conditions

### Continuous Improvement

Monthly review:
1. Calculate performance metrics
2. Identify common errors
3. Adjust parameters
4. A/B test changes
5. Document results

## Troubleshooting

### Low Signal Generation

**Problem**: Not finding trades

**Solutions**:
- Expand watchlist
- Adjust delta ranges
- Lower IV rank minimum
- Check liquidity filters

### High Slippage

**Problem**: Poor fill prices

**Solutions**:
- Use limit orders only
- Increase liquidity requirements
- Avoid market orders
- Trade during high volume hours

### Frequent Stop Losses

**Problem**: Too many losing trades

**Solutions**:
- Increase delta buffer (more OTM)
- Better entry timing (higher IV rank)
- Earlier exits (tighter stops)
- Smaller position size

### Low Returns

**Problem**: Not making enough profit

**Solutions**:
- Increase position size (within risk limits)
- Optimize take profit level
- Add more strategies
- Expand to more symbols

## Resources

### Recommended Reading
- "Options as a Strategic Investment" - Lawrence McMillan
- "Option Volatility and Pricing" - Sheldon Natenberg
- "The Options Playbook" - Brian Overby

### Online Resources
- tastytrade.com - Options education
- OptionAlpha.com - Strategy guides
- CBOE - Options data and education

### Tools
- Think or Swim - Analysis platform
- OptionStrat.com - Strategy visualization
- Market Chameleon - Options flow data

---

**Remember**: No strategy works all the time. Diversify, manage risk, and continuously learn! 📈


