#!/bin/bash
# Check health of trading agent running on remote server

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get server IP
if [ -z "$1" ]; then
    read -p "Enter server IP address: " SERVER_IP
else
    SERVER_IP=$1
fi

read -p "Enter SSH user (default: trader): " SSH_USER
SSH_USER=${SSH_USER:-trader}

echo ""
echo "========================================="
echo "Remote Health Check: $SERVER_IP"
echo "========================================="
echo ""

# Check if server is reachable
echo -n "Testing connection... "
if ping -c 1 -W 2 $SERVER_IP > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server reachable${NC}"
else
    echo -e "${RED}❌ Server unreachable${NC}"
    exit 1
fi

# SSH and run health checks
ssh ${SSH_USER}@${SERVER_IP} << 'EOFREMOTE'
#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=== SYSTEM STATUS ==="
echo ""

# Check if service is running
echo -n "Trading Agent Service: "
if systemctl is-active --quiet trading-agent; then
    echo -e "${GREEN}✅ Running${NC}"
    UPTIME=$(systemctl show trading-agent --property=ActiveEnterTimestamp | cut -d'=' -f2)
    echo "  Started: $UPTIME"
else
    echo -e "${RED}❌ Not running${NC}"
    echo ""
    echo "Recent errors:"
    sudo journalctl -u trading-agent -n 10 --no-pager
    exit 1
fi

echo ""
echo "=== RESOURCE USAGE ==="
echo ""

# CPU and Memory
echo "CPU & Memory:"
echo "$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "  CPU Usage: " 100 - $1 "%"}')"
echo "$(free -h | awk '/^Mem:/ {print "  Memory: " $3 "/" $2 " (" int($3/$2 * 100) "%)"}')"
echo "$(free -h | awk '/^Swap:/ {print "  Swap: " $3 "/" $2}')"

# Disk space
echo ""
echo "Disk Space:"
df -h / | awk 'NR==2 {print "  Root: " $3 "/" $2 " (" $5 ")"}'

echo ""
echo "=== AGENT STATUS ==="
echo ""

# Check data collection
cd ~/Automated-Options-Trading-Agent 2>/dev/null || cd /home/trader/Automated-Options-Trading-Agent

if [ -f "logs/trading_agent.log" ]; then
    LAST_LINE=$(tail -1 logs/trading_agent.log)
    LAST_TIME=$(echo "$LAST_LINE" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
    
    if [ -n "$LAST_TIME" ]; then
        echo "Latest Log Entry:"
        echo "  Time: $LAST_TIME"
        echo "  $(echo $LAST_LINE | cut -c1-80)"
        
        # Check if recent (within 5 minutes)
        CURRENT_TIME=$(date +%s)
        LOG_TIME=$(date -d "$LAST_TIME" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$LAST_TIME" +%s 2>/dev/null)
        if [ -n "$LOG_TIME" ]; then
            DIFF=$((CURRENT_TIME - LOG_TIME))
            if [ $DIFF -lt 300 ]; then
                echo -e "  Status: ${GREEN}✅ Active (${DIFF}s ago)${NC}"
            else
                echo -e "  Status: ${YELLOW}⚠️  Stale (${DIFF}s ago)${NC}"
            fi
        fi
    fi
    
    # Check for errors
    echo ""
    ERROR_COUNT=$(grep -i "error" logs/trading_agent.log 2>/dev/null | tail -100 | wc -l)
    echo "Recent Errors: $ERROR_COUNT (last 100 lines)"
    
    if [ $ERROR_COUNT -gt 0 ]; then
        echo ""
        echo "Latest errors:"
        grep -i "error" logs/trading_agent.log | tail -3
    fi
fi

# Database check
echo ""
echo "=== DATABASE ==="
echo ""

if command -v psql > /dev/null; then
    echo -n "PostgreSQL: "
    if systemctl is-active --quiet postgresql; then
        echo -e "${GREEN}✅ Running${NC}"
        
        # Check database size
        DB_SIZE=$(psql -U trader -d trading_agent -t -c "SELECT pg_size_pretty(pg_database_size('trading_agent'));" 2>/dev/null | tr -d ' ')
        if [ -n "$DB_SIZE" ]; then
            echo "  Database Size: $DB_SIZE"
        fi
        
        # Check tick data count
        TICK_COUNT=$(psql -U trader -d trading_agent -t -c "SELECT COUNT(*) FROM index_tick_data;" 2>/dev/null | tr -d ' ')
        if [ -n "$TICK_COUNT" ]; then
            echo "  Tick Records: $TICK_COUNT"
        fi
    else
        echo -e "${RED}❌ Not running${NC}"
    fi
fi

echo ""
echo "=== NETWORK ==="
echo ""

# Check internet connectivity
echo -n "Internet: "
if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${RED}❌ No connection${NC}"
fi

# Check Alpaca API
echo -n "Alpaca API: "
if curl -s --max-time 5 https://paper-api.alpaca.markets/v2/clock > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Reachable${NC}"
else
    echo -e "${RED}❌ Unreachable${NC}"
fi

echo ""
echo "========================================="
echo "Health check complete!"
echo "========================================="
echo ""

EOFREMOTE

echo ""
echo "To view logs, run:"
echo "  ssh ${SSH_USER}@${SERVER_IP} 'tail -f ~/Automated-Options-Trading-Agent/logs/trading_agent.log'"
echo ""

