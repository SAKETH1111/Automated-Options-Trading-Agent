#!/usr/bin/env python3
"""
Trading Agent Dashboard
Simple web dashboard to monitor the trading agent status
"""

import os
import json
import subprocess
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/status':
            self.serve_api_status()
        elif self.path == '/api/data':
            self.serve_api_data()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Agent Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric { text-align: center; padding: 15px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .metric-label { color: #666; margin-top: 5px; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
        .timestamp { color: #666; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Trading Agent Dashboard</h1>
        <p class="timestamp">Last updated: <span id="timestamp"></span></p>
        
        <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Service Status</h3>
                <div id="service-status">Loading...</div>
            </div>
            
            <div class="card">
                <h3>📈 Data Collection</h3>
                <div id="data-collection">Loading...</div>
            </div>
            
            <div class="card">
                <h3>💻 System Resources</h3>
                <div id="system-resources">Loading...</div>
            </div>
            
            <div class="card">
                <h3>💰 Latest Prices</h3>
                <div id="latest-prices">Loading...</div>
            </div>
        </div>
        
        <div class="card">
            <h3>📋 Recent Activity</h3>
            <div id="recent-activity">Loading...</div>
        </div>
    </div>
    
    <script>
        function updateTimestamp() {
            document.getElementById('timestamp').textContent = new Date().toLocaleString();
        }
        
        async function loadData() {
            updateTimestamp();
            
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // Update service status
                const serviceStatus = document.getElementById('service-status');
                if (data.service === 'active') {
                    serviceStatus.innerHTML = '<div class="status success">✅ Trading Agent is RUNNING</div>';
                } else {
                    serviceStatus.innerHTML = '<div class="status error">❌ Trading Agent is NOT RUNNING</div>';
                }
                
                // Update data collection
                const dataCollection = document.getElementById('data-collection');
                const totalTicks = data.total_ticks || 0;
                const recentTicks = data.recent_ticks || 0;
                dataCollection.innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${totalTicks}</div>
                        <div class="metric-label">Total Ticks Collected</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${recentTicks}</div>
                        <div class="metric-label">Ticks (Last 5 min)</div>
                    </div>
                `;
                
                // Update system resources
                const systemResources = document.getElementById('system-resources');
                systemResources.innerHTML = `
                    <div class="metric">
                        <div class="metric-value">${data.memory_usage || 'N/A'}</div>
                        <div class="metric-label">Memory Usage</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${data.cpu_usage || 'N/A'}</div>
                        <div class="metric-label">CPU Usage</div>
                    </div>
                `;
                
                // Update latest prices
                const latestPrices = document.getElementById('latest-prices');
                if (data.latest_prices && data.latest_prices.length > 0) {
                    let pricesHtml = '<table><tr><th>Symbol</th><th>Price</th><th>Time</th></tr>';
                    data.latest_prices.forEach(price => {
                        pricesHtml += `<tr><td>${price.symbol}</td><td>$${price.price}</td><td>${price.timestamp}</td></tr>`;
                    });
                    pricesHtml += '</table>';
                    latestPrices.innerHTML = pricesHtml;
                } else {
                    latestPrices.innerHTML = '<div class="status warning">No recent price data</div>';
                }
                
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('service-status').innerHTML = '<div class="status error">❌ Error loading status</div>';
            }
        }
        
        // Load data on page load and refresh every 30 seconds
        loadData();
        setInterval(loadData, 30000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_api_status(self):
        try:
            # Get service status
            result = subprocess.run(['systemctl', 'is-active', 'trading-agent'], 
                                  capture_output=True, text=True)
            service_status = result.stdout.strip()
            
            # Get data counts
            result = subprocess.run([
                'sudo', '-u', 'postgres', 'psql', '-d', 'options_trading', 
                '-c', "SELECT COUNT(*) FROM index_tick_data;"
            ], capture_output=True, text=True)
            total_ticks = 0
            if result.returncode == 0:
                total_ticks = int(result.stdout.split('\n')[2].strip() or 0)
            
            # Get recent data count
            result = subprocess.run([
                'sudo', '-u', 'postgres', 'psql', '-d', 'options_trading', 
                '-c', "SELECT COUNT(*) FROM index_tick_data WHERE timestamp > NOW() - INTERVAL '5 minutes';"
            ], capture_output=True, text=True)
            recent_ticks = 0
            if result.returncode == 0:
                recent_ticks = int(result.stdout.split('\n')[2].strip() or 0)
            
            # Get latest prices
            result = subprocess.run([
                'sudo', '-u', 'postgres', 'psql', '-d', 'options_trading', 
                '-c', "SELECT symbol, price, timestamp FROM index_tick_data ORDER BY timestamp DESC LIMIT 4;"
            ], capture_output=True, text=True)
            
            latest_prices = []
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[2:-2]:  # Skip header and footer
                    if line.strip():
                        parts = line.split('|')
                        if len(parts) >= 3:
                            latest_prices.append({
                                'symbol': parts[0].strip(),
                                'price': parts[1].strip(),
                                'timestamp': parts[2].strip()
                            })
            
            # Get memory usage
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            memory_usage = 'N/A'
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'python main.py' in line:
                        parts = line.split()
                        if len(parts) > 5:
                            memory_kb = int(parts[5])
                            memory_usage = f"{memory_kb / 1024:.1f} MB"
                        break
            
            response = {
                'service': service_status,
                'total_ticks': total_ticks,
                'recent_ticks': recent_ticks,
                'latest_prices': latest_prices,
                'memory_usage': memory_usage,
                'timestamp': datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def serve_api_data(self):
        # This could be expanded to serve more detailed data
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'message': 'Data API endpoint'}).encode())
    
    def log_message(self, format, *args):
        pass

def main():
    port = int(os.environ.get('DASHBOARD_PORT', 8081))
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"📊 Dashboard server starting on port {port}")
    print(f"🌐 Dashboard URL: http://45.55.150.19:{port}")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard server stopped")
        server.shutdown()

if __name__ == '__main__':
    main()
