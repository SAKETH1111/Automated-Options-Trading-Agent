#!/bin/bash

# Trading Agent Log & Data Monitor
# Usage: ./monitor_logs.sh [logs|data|trades|dashboard]

SERVER_IP="45.55.150.19"

show_help() {
    echo "🔍 Trading Agent Monitor"
    echo "======================="
    echo ""
    echo "Usage: ./monitor_logs.sh [command]"
    echo ""
    echo "Commands:"
    echo "  logs      - Show recent logs"
    echo "  live      - Show live logs (real-time)"
    echo "  data      - Show data collection stats"
    echo "  trades    - Show recent price data"
    echo "  dashboard - Open web dashboard"
    echo "  status    - Show system status"
    echo "  errors    - Show error logs only"
    echo "  help      - Show this help"
    echo ""
    echo "Examples:"
    echo "  ./monitor_logs.sh logs      # Show last 20 log entries"
    echo "  ./monitor_logs.sh live      # Watch logs in real-time"
    echo "  ./monitor_logs.sh data      # Show data collection stats"
    echo "  ./monitor_logs.sh dashboard # Open web dashboard"
}

show_logs() {
    echo "📋 Recent Logs (Last 20 entries):"
    echo "=================================="
    ssh root@$SERVER_IP "journalctl -u trading-agent --no-pager -n 20"
}

show_live_logs() {
    echo "📺 Live Logs (Press Ctrl+C to stop):"
    echo "====================================="
    ssh root@$SERVER_IP "journalctl -u trading-agent -f"
}

show_data() {
    echo "📊 Data Collection Statistics:"
    echo "=============================="
    
    # Total data count
    echo "Total Ticks Collected:"
    ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT COUNT(*) as total_ticks FROM index_tick_data;\"" 2>/dev/null
    
    echo ""
    echo "Data by Symbol:"
    ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT symbol, COUNT(*) as ticks, MIN(timestamp) as first_data, MAX(timestamp) as latest_data FROM index_tick_data GROUP BY symbol ORDER BY ticks DESC;\"" 2>/dev/null
    
    echo ""
    echo "Recent Activity (Last 5 minutes):"
    ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT COUNT(*) as recent_ticks FROM index_tick_data WHERE timestamp > NOW() - INTERVAL '5 minutes';\"" 2>/dev/null
}

show_trades() {
    echo "💰 Recent Price Data:"
    echo "===================="
    ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT symbol, price, volume, timestamp FROM index_tick_data ORDER BY timestamp DESC LIMIT 15;\"" 2>/dev/null
}

show_dashboard() {
    echo "🌐 Opening Web Dashboard..."
    echo "Dashboard URL: http://$SERVER_IP:8081"
    echo ""
    echo "If the dashboard doesn't open automatically:"
    echo "1. Open your browser"
    echo "2. Go to: http://$SERVER_IP:8081"
    echo ""
    
    # Try to open in browser (works on macOS)
    if command -v open &> /dev/null; then
        open "http://$SERVER_IP:8081"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "http://$SERVER_IP:8081"
    else
        echo "Please manually open: http://$SERVER_IP:8081"
    fi
}

show_status() {
    echo "🔍 System Status:"
    echo "================"
    
    # Service status
    echo "Service Status:"
    ssh root@$SERVER_IP "systemctl is-active trading-agent"
    
    echo ""
    echo "Memory Usage:"
    ssh root@$SERVER_IP "ps aux | grep 'python main.py' | grep -v grep | awk '{print \$6/1024 \"MB\"}'"
    
    echo ""
    echo "Recent Data Count:"
    ssh root@$SERVER_IP "sudo -u postgres psql -d options_trading -c \"SELECT COUNT(*) FROM index_tick_data WHERE timestamp > NOW() - INTERVAL '5 minutes';\"" 2>/dev/null | grep -o '[0-9]*' | tail -1
}

show_errors() {
    echo "❌ Error Logs (Last 10 errors):"
    echo "==============================="
    ssh root@$SERVER_IP "journalctl -u trading-agent -p err --no-pager -n 10"
}

# Main script logic
case "${1:-help}" in
    "logs")
        show_logs
        ;;
    "live")
        show_live_logs
        ;;
    "data")
        show_data
        ;;
    "trades")
        show_trades
        ;;
    "dashboard")
        show_dashboard
        ;;
    "status")
        show_status
        ;;
    "errors")
        show_errors
        ;;
    "help"|*)
        show_help
        ;;
esac
