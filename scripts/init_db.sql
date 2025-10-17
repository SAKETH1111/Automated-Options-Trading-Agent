-- Database Initialization Script for Production Deployment
-- Institutional Options Trading System

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create custom types
CREATE TYPE option_type AS ENUM ('call', 'put');
CREATE TYPE trade_action AS ENUM ('buy', 'sell', 'open', 'close');
CREATE TYPE strategy_type AS ENUM ('bull_put_spread', 'cash_secured_put', 'bear_call_spread', 'iron_condor', 'calendar_spread', 'diagonal_spread');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected');
CREATE TYPE alert_severity AS ENUM ('critical', 'high', 'normal', 'low');

-- Create tables for trading data
CREATE TABLE IF NOT EXISTS trade_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    size INTEGER NOT NULL,
    exchange VARCHAR(10),
    conditions TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS quote_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    bid DECIMAL(10,4) NOT NULL,
    ask DECIMAL(10,4) NOT NULL,
    bid_size INTEGER NOT NULL,
    ask_size INTEGER NOT NULL,
    exchange VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS options_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    contract_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    last DECIMAL(10,4),
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility DECIMAL(8,6),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    rho DECIMAL(8,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for trading system
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id VARCHAR(50) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    contract_type option_type NOT NULL,
    strike DECIMAL(10,4) NOT NULL,
    expiration DATE NOT NULL,
    quantity INTEGER NOT NULL,
    entry_price DECIMAL(10,4) NOT NULL,
    current_price DECIMAL(10,4),
    unrealized_pnl DECIMAL(10,2),
    realized_pnl DECIMAL(10,2) DEFAULT 0,
    strategy_type strategy_type,
    entry_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_timestamp TIMESTAMP WITH TIME ZONE,
    is_closed BOOLEAN DEFAULT FALSE,
    is_day_trade BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trade_id VARCHAR(50) UNIQUE NOT NULL,
    position_id VARCHAR(50) REFERENCES positions(position_id),
    symbol VARCHAR(20) NOT NULL,
    contract_type option_type NOT NULL,
    strike DECIMAL(10,4) NOT NULL,
    expiration DATE NOT NULL,
    action trade_action NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    commission DECIMAL(10,2) DEFAULT 0,
    strategy_type strategy_type,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    order_status order_status DEFAULT 'filled',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    account_value DECIMAL(15,2) NOT NULL,
    daily_pnl DECIMAL(10,2) NOT NULL,
    unrealized_pnl DECIMAL(10,2) NOT NULL,
    realized_pnl DECIMAL(10,2) NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    portfolio_heat DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    portfolio_delta DECIMAL(10,2),
    portfolio_gamma DECIMAL(10,2),
    portfolio_theta DECIMAL(10,2),
    portfolio_vega DECIMAL(10,2),
    portfolio_rho DECIMAL(10,2),
    var_95 DECIMAL(10,2),
    cvar_95 DECIMAL(10,2),
    max_loss_per_trade DECIMAL(10,2),
    daily_loss_limit DECIMAL(10,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS circuit_breakers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    breaker_type VARCHAR(50) NOT NULL,
    breaker_level VARCHAR(50) NOT NULL,
    threshold_value DECIMAL(15,4) NOT NULL,
    current_value DECIMAL(15,4) NOT NULL,
    is_triggered BOOLEAN DEFAULT FALSE,
    triggered_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity alert_severity NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(100),
    channels TEXT[] NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    prediction_value DECIMAL(15,6) NOT NULL,
    confidence DECIMAL(5,4),
    features JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_regimes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regime_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    vix_level DECIMAL(8,4),
    market_conditions JSONB,
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS volatility_surfaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    expiration DATE NOT NULL,
    strike DECIMAL(10,4) NOT NULL,
    option_type option_type NOT NULL,
    implied_volatility DECIMAL(8,6) NOT NULL,
    bid DECIMAL(10,4),
    ask DECIMAL(10,4),
    volume INTEGER,
    open_interest INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, expiration, strike, option_type, timestamp)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trade_data_symbol_timestamp ON trade_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_data_timestamp ON trade_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_quote_data_symbol_timestamp ON quote_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_quote_data_timestamp ON quote_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_options_data_symbol_timestamp ON options_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_options_data_contract_id ON options_data(contract_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_entry_timestamp ON positions(entry_timestamp);
CREATE INDEX IF NOT EXISTS idx_positions_is_closed ON positions(is_closed);
CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_type ON trades(strategy_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_timestamp ON risk_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_breaker_type ON circuit_breakers(breaker_type);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_is_triggered ON circuit_breakers(is_triggered);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_name ON ml_predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_regimes_detected_at ON market_regimes(detected_at);
CREATE INDEX IF NOT EXISTS idx_volatility_surfaces_symbol_expiration ON volatility_surfaces(symbol, expiration);
CREATE INDEX IF NOT EXISTS idx_volatility_surfaces_timestamp ON volatility_surfaces(timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW active_positions AS
SELECT 
    p.*,
    (p.current_price - p.entry_price) * p.quantity as current_pnl,
    EXTRACT(DAYS FROM (p.expiration - CURRENT_DATE)) as days_to_expiration
FROM positions p
WHERE p.is_closed = FALSE;

CREATE OR REPLACE VIEW daily_pnl_summary AS
SELECT 
    DATE(timestamp) as trade_date,
    SUM(CASE WHEN action = 'sell' THEN price * quantity ELSE -price * quantity END) as daily_pnl,
    COUNT(*) as trade_count
FROM trades
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;

CREATE OR REPLACE VIEW strategy_performance AS
SELECT 
    strategy_type,
    COUNT(*) as total_trades,
    SUM(realized_pnl) as total_pnl,
    AVG(realized_pnl) as avg_pnl,
    COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
    ROUND(
        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END)::DECIMAL / COUNT(*) * 100, 2
    ) as win_rate
FROM positions
WHERE is_closed = TRUE
GROUP BY strategy_type
ORDER BY total_pnl DESC;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_position_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER update_positions_timestamp
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_position_timestamp();

-- Create function to calculate portfolio Greeks
CREATE OR REPLACE FUNCTION calculate_portfolio_greeks()
RETURNS TABLE(
    total_delta DECIMAL(10,2),
    total_gamma DECIMAL(10,2),
    total_theta DECIMAL(10,2),
    total_vega DECIMAL(10,2),
    total_rho DECIMAL(10,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(p.quantity * o.delta), 0) as total_delta,
        COALESCE(SUM(p.quantity * o.gamma), 0) as total_gamma,
        COALESCE(SUM(p.quantity * o.theta), 0) as total_theta,
        COALESCE(SUM(p.quantity * o.vega), 0) as total_vega,
        COALESCE(SUM(p.quantity * o.rho), 0) as total_rho
    FROM positions p
    LEFT JOIN options_data o ON p.symbol = o.symbol 
        AND p.strike = o.strike 
        AND p.expiration = o.expiration
        AND p.contract_type = o.contract_type
    WHERE p.is_closed = FALSE
    AND o.timestamp = (
        SELECT MAX(timestamp) 
        FROM options_data o2 
        WHERE o2.symbol = o.symbol 
        AND o2.strike = o.strike 
        AND o2.expiration = o.expiration
        AND o2.contract_type = o.contract_type
    );
END;
$$ LANGUAGE plpgsql;

-- Create function to check PDT compliance
CREATE OR REPLACE FUNCTION check_pdt_compliance(account_value DECIMAL)
RETURNS TABLE(
    day_trades_used INTEGER,
    day_trades_remaining INTEGER,
    is_pdt_violation BOOLEAN,
    can_day_trade BOOLEAN
) AS $$
DECLARE
    day_trade_count INTEGER;
    max_day_trades INTEGER := 3;
BEGIN
    -- Count day trades in rolling 5-day window
    SELECT COUNT(*)
    INTO day_trade_count
    FROM positions
    WHERE is_day_trade = TRUE
    AND entry_timestamp >= CURRENT_DATE - INTERVAL '5 days';
    
    RETURN QUERY
    SELECT 
        day_trade_count,
        GREATEST(0, max_day_trades - day_trade_count),
        day_trade_count >= max_day_trades,
        day_trade_count < max_day_trades;
END;
$$ LANGUAGE plpgsql;

-- Insert initial data
INSERT INTO performance_metrics (date, account_value, daily_pnl, unrealized_pnl, realized_pnl)
VALUES (CURRENT_DATE, 5000.00, 0.00, 0.00, 0.00)
ON CONFLICT (date) DO NOTHING;

-- Create user for application
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'trading_user') THEN
        CREATE ROLE trading_user WITH LOGIN PASSWORD 'trading_password';
    END IF;
END
$$;

-- Grant permissions
GRANT CONNECT ON DATABASE trading_agent TO trading_user;
GRANT USAGE ON SCHEMA public TO trading_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trading_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trading_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO trading_user;

-- Create backup user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'backup_user') THEN
        CREATE ROLE backup_user WITH LOGIN PASSWORD 'backup_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE trading_agent TO backup_user;
GRANT USAGE ON SCHEMA public TO backup_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO backup_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO backup_user;

-- Create monitoring user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'monitoring_user') THEN
        CREATE ROLE monitoring_user WITH LOGIN PASSWORD 'monitoring_password';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE trading_agent TO monitoring_user;
GRANT USAGE ON SCHEMA public TO monitoring_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO monitoring_user;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name VARCHAR(50) NOT NULL,
    operation VARCHAR(10) NOT NULL,
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (table_name, operation, old_values, new_values, user_id)
    VALUES (
        TG_TABLE_NAME,
        TG_OP,
        CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE NULL END,
        CASE WHEN TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN to_jsonb(NEW) ELSE NULL END,
        current_user
    );
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for critical tables
CREATE TRIGGER audit_positions
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_trades
    AFTER INSERT OR UPDATE OR DELETE ON trades
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_circuit_breakers
    AFTER INSERT OR UPDATE OR DELETE ON circuit_breakers
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create performance monitoring views
CREATE OR REPLACE VIEW system_performance AS
SELECT 
    'Database Size' as metric,
    pg_size_pretty(pg_database_size(current_database())) as value
UNION ALL
SELECT 
    'Active Connections',
    count(*)::text
FROM pg_stat_activity
WHERE state = 'active'
UNION ALL
SELECT 
    'Total Tables',
    count(*)::text
FROM information_schema.tables
WHERE table_schema = 'public'
UNION ALL
SELECT 
    'Total Positions',
    count(*)::text
FROM positions
UNION ALL
SELECT 
    'Active Positions',
    count(*)::text
FROM positions
WHERE is_closed = FALSE;

-- Create maintenance procedures
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete old trade data (keep 1 year)
    DELETE FROM trade_data 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 year';
    
    -- Delete old quote data (keep 1 year)
    DELETE FROM quote_data 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 year';
    
    -- Delete old options data (keep 1 year)
    DELETE FROM options_data 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '1 year';
    
    -- Delete old audit logs (keep 7 years)
    DELETE FROM audit_log 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '7 years';
    
    -- Update statistics
    ANALYZE;
END;
$$ LANGUAGE plpgsql;

-- Create index maintenance function
CREATE OR REPLACE FUNCTION maintain_indexes()
RETURNS void AS $$
BEGIN
    -- Reindex all tables
    REINDEX DATABASE trading_agent;
    
    -- Update statistics
    ANALYZE;
END;
$$ LANGUAGE plpgsql;

-- Create backup function
CREATE OR REPLACE FUNCTION create_backup()
RETURNS void AS $$
BEGIN
    -- This would typically call pg_dump
    -- For now, just log the request
    INSERT INTO audit_log (table_name, operation, new_values, user_id)
    VALUES ('backup', 'CREATE', '{"backup_requested": true}', current_user);
END;
$$ LANGUAGE plpgsql;

-- Final setup
COMMENT ON DATABASE trading_agent IS 'Institutional Options Trading System Database';
COMMENT ON TABLE positions IS 'Active and closed positions';
COMMENT ON TABLE trades IS 'All trade executions';
COMMENT ON TABLE performance_metrics IS 'Daily performance metrics';
COMMENT ON TABLE circuit_breakers IS 'Circuit breaker triggers and status';
COMMENT ON TABLE alerts IS 'System alerts and notifications';
COMMENT ON TABLE ml_predictions IS 'ML model predictions';
COMMENT ON TABLE volatility_surfaces IS 'Options volatility surface data';

-- Create initial configuration
INSERT INTO circuit_breakers (breaker_type, breaker_level, threshold_value, current_value, description)
VALUES 
    ('position', 'stop_loss', 2.0, 0.0, 'Stop loss at 2x credit received'),
    ('position', 'take_profit', 0.5, 0.0, 'Take profit at 50% max profit'),
    ('portfolio', 'daily_loss', 3.0, 0.0, 'Daily loss limit at 3%'),
    ('portfolio', 'max_heat', 0.15, 0.0, 'Maximum portfolio heat at 15%'),
    ('system', 'api_errors', 10.0, 0.0, 'Maximum API errors per hour'),
    ('system', 'data_latency', 5000.0, 0.0, 'Maximum data latency in milliseconds');

-- Create initial alert
INSERT INTO alerts (alert_type, severity, title, message, channels)
VALUES ('system', 'normal', 'Database Initialized', 'Trading system database has been successfully initialized', ARRAY['telegram']);

-- Final commit
COMMIT;
