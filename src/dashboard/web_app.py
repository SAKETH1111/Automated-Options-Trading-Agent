"""
Advanced Web Dashboard
Real-time trading dashboard with FastAPI
"""

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List
from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session

from src.database.session import get_session
from src.database.models import Trade, IndexTickData, TechnicalIndicators
from src.automation.performance_tracker import PerformanceTracker
from src.risk_management.portfolio_risk import PortfolioRiskManager

app = FastAPI(title="Trading Agent Dashboard")

# HTML Dashboard
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Agent Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #e2e8f0; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .subtitle { opacity: 0.9; font-size: 1.1em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #1e293b; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h3 { color: #a78bfa; margin-bottom: 15px; font-size: 1.3em; }
        .metric { text-align: center; padding: 20px; }
        .metric-value { font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }
        .metric-value.positive { color: #10b981; }
        .metric-value.negative { color: #ef4444; }
        .metric-value.neutral { color: #60a5fa; }
        .metric-label { color: #94a3b8; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }
        .status { padding: 8px 16px; border-radius: 20px; display: inline-block; font-size: 0.9em; font-weight: 600; }
        .status.active { background: #10b981; color: white; }
        .status.paused { background: #ef4444; color: white; }
        .status.warning { background: #f59e0b; color: white; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #334155; color: #a78bfa; font-weight: 600; }
        tr:hover { background: #334155; }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 1em; font-weight: 600; }
        .refresh-btn:hover { background: #5568d3; }
        .chart { min-height: 400px; margin-top: 20px; }
        #timestamp { color: #94a3b8; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Trading Agent Dashboard</h1>
            <div class="subtitle">Real-time monitoring and performance tracking</div>
            <div id="timestamp" style="margin-top: 10px;"></div>
        </div>
        
        <button class="refresh-btn" onclick="loadData()">🔄 Refresh Data</button>
        
        <div class="grid" style="margin-top: 20px;">
            <div class="card">
                <h3>💰 Account Summary</h3>
                <div id="account-summary">Loading...</div>
            </div>
            
            <div class="card">
                <h3>📊 Performance</h3>
                <div id="performance">Loading...</div>
            </div>
            
            <div class="card">
                <h3>🎯 Open Positions</h3>
                <div id="positions">Loading...</div>
            </div>
            
            <div class="card">
                <h3>🛡️ Risk Status</h3>
                <div id="risk-status">Loading...</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Equity Curve</h3>
            <div id="equity-chart" class="chart"></div>
        </div>
        
        <div class="card">
            <h3>📋 Recent Trades</h3>
            <div id="recent-trades">Loading...</div>
        </div>
        
        <div class="card">
            <h3>📊 Price Chart</h3>
            <div id="price-chart" class="chart"></div>
        </div>
    </div>
    
    <script>
        function updateTimestamp() {
            document.getElementById('timestamp').textContent = 
                'Last updated: ' + new Date().toLocaleString();
        }
        
        async function loadData() {
            updateTimestamp();
            
            try {
                // Load all data
                await Promise.all([
                    loadAccountSummary(),
                    loadPerformance(),
                    loadPositions(),
                    loadRiskStatus(),
                    loadRecentTrades(),
                    loadCharts()
                ]);
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        async function loadAccountSummary() {
            const response = await fetch('/api/account');
            const data = await response.json();
            
            document.getElementById('account-summary').innerHTML = `
                <div class="metric">
                    <div class="metric-value neutral">$${data.equity.toLocaleString()}</div>
                    <div class="metric-label">Total Equity</div>
                </div>
                <div class="metric">
                    <div class="metric-value neutral">$${data.cash.toLocaleString()}</div>
                    <div class="metric-label">Cash Available</div>
                </div>
            `;
        }
        
        async function loadPerformance() {
            const response = await fetch('/api/performance');
            const data = await response.json();
            
            const pnlClass = data.total_pnl >= 0 ? 'positive' : 'negative';
            
            document.getElementById('performance').innerHTML = `
                <div class="metric">
                    <div class="metric-value ${pnlClass}">$${data.total_pnl.toLocaleString()}</div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric">
                    <div class="metric-value neutral">${(data.win_rate * 100).toFixed(1)}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            `;
        }
        
        async function loadPositions() {
            const response = await fetch('/api/positions');
            const data = await response.json();
            
            document.getElementById('positions').innerHTML = `
                <div class="metric">
                    <div class="metric-value neutral">${data.total}</div>
                    <div class="metric-label">Open Positions</div>
                </div>
                <div class="metric">
                    <div class="metric-value ${data.current_pnl >= 0 ? 'positive' : 'negative'}">
                        $${data.current_pnl.toLocaleString()}
                    </div>
                    <div class="metric-label">Current P&L</div>
                </div>
            `;
        }
        
        async function loadRiskStatus() {
            const response = await fetch('/api/risk');
            const data = await response.json();
            
            const statusClass = data.circuit_breaker_tripped ? 'paused' : 'active';
            const statusText = data.circuit_breaker_tripped ? 'PAUSED' : 'ACTIVE';
            
            document.getElementById('risk-status').innerHTML = `
                <div style="text-align: center; margin-bottom: 20px;">
                    <span class="status ${statusClass}">${statusText}</span>
                </div>
                <div class="metric">
                    <div class="metric-value neutral">${data.total_risk_pct.toFixed(1)}%</div>
                    <div class="metric-label">Portfolio Risk</div>
                </div>
            `;
        }
        
        async function loadRecentTrades() {
            const response = await fetch('/api/trades/recent');
            const data = await response.json();
            
            if (data.trades.length === 0) {
                document.getElementById('recent-trades').innerHTML = '<p style="text-align:center;color:#94a3b8;">No recent trades</p>';
                return;
            }
            
            let html = '<table><tr><th>Date</th><th>Symbol</th><th>Strategy</th><th>P&L</th><th>Status</th></tr>';
            
            data.trades.forEach(trade => {
                const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
                html += `<tr>
                    <td>${new Date(trade.date).toLocaleDateString()}</td>
                    <td>${trade.symbol}</td>
                    <td>${trade.strategy}</td>
                    <td style="color: ${trade.pnl >= 0 ? '#10b981' : '#ef4444'}">$${trade.pnl.toFixed(2)}</td>
                    <td>${trade.status}</td>
                </tr>`;
            });
            
            html += '</table>';
            document.getElementById('recent-trades').innerHTML = html;
        }
        
        async function loadCharts() {
            // Load equity curve
            const equityResponse = await fetch('/api/charts/equity');
            const equityData = await equityResponse.json();
            
            Plotly.newPlot('equity-chart', [{
                x: equityData.dates,
                y: equityData.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Equity',
                line: { color: '#667eea', width: 2 }
            }], {
                title: 'Equity Curve',
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                xaxis: { gridcolor: '#334155' },
                yaxis: { gridcolor: '#334155' }
            });
            
            // Load price chart
            const priceResponse = await fetch('/api/charts/price?symbol=SPY');
            const priceData = await priceResponse.json();
            
            Plotly.newPlot('price-chart', [{
                x: priceData.dates,
                y: priceData.prices,
                type: 'scatter',
                mode: 'lines',
                name: 'SPY Price',
                line: { color: '#10b981', width: 2 }
            }], {
                title: 'SPY Price (Last 24 Hours)',
                paper_bgcolor: '#1e293b',
                plot_bgcolor: '#1e293b',
                font: { color: '#e2e8f0' },
                xaxis: { gridcolor: '#334155' },
                yaxis: { gridcolor: '#334155' }
            });
        }
        
        // Load data on page load and refresh every 30 seconds
        loadData();
        setInterval(loadData, 30000);
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    """Serve dashboard HTML"""
    return HTMLResponse(content=dashboard_html)

@app.get("/api/account")
async def get_account():
    """Get account summary"""
    # Placeholder - integrate with your Alpaca client
    return {
        "equity": 10000.0,
        "cash": 8500.0,
        "buying_power": 8500.0
    }

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    db = get_session()
    tracker = PerformanceTracker(db)
    
    all_time = tracker.get_all_time_stats()
    
    db.close()
    
    return {
        "total_pnl": all_time.get('total_pnl', 0),
        "win_rate": all_time.get('win_rate', 0),
        "total_trades": all_time.get('total_trades', 0)
    }

@app.get("/api/positions")
async def get_positions():
    """Get open positions"""
    db = get_session()
    
    open_trades = db.query(Trade).filter(Trade.status == 'open').all()
    current_pnl = sum(t.pnl for t in open_trades)
    
    db.close()
    
    return {
        "total": len(open_trades),
        "current_pnl": current_pnl
    }

@app.get("/api/risk")
async def get_risk_status():
    """Get risk status"""
    db = get_session()
    risk_manager = PortfolioRiskManager(db, total_capital=10000.0)
    
    metrics = risk_manager.get_portfolio_risk_metrics()
    
    db.close()
    
    return {
        "total_risk_pct": metrics.get('total_risk_pct', 0),
        "circuit_breaker_tripped": False  # Check actual circuit breaker
    }

@app.get("/api/trades/recent")
async def get_recent_trades():
    """Get recent trades"""
    db = get_session()
    
    recent = db.query(Trade).filter(
        Trade.status == 'closed'
    ).order_by(Trade.timestamp_exit.desc()).limit(10).all()
    
    trades = [{
        "date": t.timestamp_exit.isoformat() if t.timestamp_exit else None,
        "symbol": t.symbol,
        "strategy": t.strategy,
        "pnl": t.pnl,
        "status": t.status
    } for t in recent]
    
    db.close()
    
    return {"trades": trades}

@app.get("/api/charts/equity")
async def get_equity_chart():
    """Get equity curve data"""
    # Placeholder - calculate from trades
    return {
        "dates": [(datetime.now() - timedelta(days=i)).isoformat() for i in range(30, 0, -1)],
        "values": [10000 + i * 50 for i in range(30)]
    }

@app.get("/api/charts/price")
async def get_price_chart(symbol: str = "SPY"):
    """Get price chart data"""
    db = get_session()
    
    cutoff = datetime.utcnow() - timedelta(hours=24)
    data = db.query(IndexTickData).filter(
        IndexTickData.symbol == symbol,
        IndexTickData.timestamp >= cutoff
    ).order_by(IndexTickData.timestamp.asc()).limit(1000).all()
    
    db.close()
    
    return {
        "dates": [d.timestamp.isoformat() for d in data],
        "prices": [d.price for d in data]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

