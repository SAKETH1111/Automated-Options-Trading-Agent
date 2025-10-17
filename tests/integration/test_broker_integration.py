"""
Integration Tests for Broker Integration
Comprehensive testing of Alpaca broker integration and order flow
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.brokers.alpaca_connector import (
    AlpacaConnector,
    OrderType,
    OrderSide,
    OrderTimeInForce,
    OptionsOrder,
    AccountInfo,
    Position
)

class TestAlpacaIntegration:
    """Test suite for Alpaca broker integration"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'broker': {
                'name': 'alpaca',
                'api_key': 'test_api_key',
                'secret_key': 'test_secret_key',
                'paper_trading': True,
                'base_url': 'https://paper-api.alpaca.markets',
                'data_url': 'https://data.alpaca.markets',
                'commission_per_contract': 0.65,
                'regulatory_fees': 0.000119
            }
        }
    
    @pytest.fixture
    def alpaca_connector(self, mock_config):
        """Create Alpaca connector with mocked config"""
        with patch('src.brokers.alpaca_connector.yaml.safe_load', return_value=mock_config):
            return AlpacaConnector()
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, alpaca_connector):
        """Test successful authentication"""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            'id': 'test_account_id',
            'buying_power': '10000.00',
            'cash': '5000.00',
            'portfolio_value': '10000.00'
        })
        
        with patch.object(alpaca_connector.session, 'get', return_value=mock_response):
            result = await alpaca_connector._test_authentication()
            assert result['id'] == 'test_account_id'
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, alpaca_connector):
        """Test authentication failure"""
        mock_response = Mock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='Unauthorized')
        
        with patch.object(alpaca_connector.session, 'get', return_value=mock_response):
            with pytest.raises(Exception, match="Authentication failed"):
                await alpaca_connector._test_authentication()
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, alpaca_connector):
        """Test getting account information"""
        mock_account_data = {
            'id': 'test_account',
            'buying_power': '10000.00',
            'cash': '5000.00',
            'portfolio_value': '10000.00',
            'equity': '10000.00',
            'long_market_value': '5000.00',
            'short_market_value': '0.00',
            'initial_margin': '2000.00',
            'maintenance_margin': '1000.00',
            'last_equity': '10000.00',
            'last_maintenance_margin': '1000.00',
            'day_trade_count': 2,
            'day_trading_buying_power': '20000.00',
            'regt_buying_power': '10000.00',
            'crypto_buying_power': '0.00',
            'non_marginable_buying_power': '5000.00'
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_account_data)
        
        with patch.object(alpaca_connector.session, 'get', return_value=mock_response):
            account_info = await alpaca_connector.get_account_info()
            
            assert isinstance(account_info, AccountInfo)
            assert account_info.account_id == 'test_account'
            assert account_info.buying_power == 10000.0
            assert account_info.day_trade_count == 2
    
    @pytest.mark.asyncio
    async def test_get_positions(self, alpaca_connector):
        """Test getting positions"""
        mock_positions_data = [
            {
                'asset_id': 'test_asset_1',
                'symbol': 'SPY',
                'exchange': 'ARCA',
                'asset_class': 'us_equity',
                'qty': '10',
                'side': 'long',
                'market_value': '4250.00',
                'cost_basis': '4200.00',
                'unrealized_pl': '50.00',
                'unrealized_plpc': '0.012',
                'unrealized_intraday_pl': '25.00',
                'unrealized_intraday_plpc': '0.006',
                'current_price': '425.00',
                'lastday_price': '420.00',
                'change_today': '5.00',
                'qty_available': '10',
                'avg_entry_price': '420.00'
            }
        ]
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_positions_data)
        
        with patch.object(alpaca_connector.session, 'get', return_value=mock_response):
            positions = await alpaca_connector.get_positions()
            
            assert len(positions) == 1
            assert isinstance(positions[0], Position)
            assert positions[0].symbol == 'SPY'
            assert positions[0].qty == 10.0
            assert positions[0].unrealized_pl == 50.0
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, alpaca_connector):
        """Test successful order placement"""
        mock_order_data = {
            'id': 'test_order_id',
            'client_order_id': 'test_client_id',
            'created_at': '2024-01-15T10:00:00Z',
            'updated_at': '2024-01-15T10:00:00Z',
            'submitted_at': '2024-01-15T10:00:00Z',
            'filled_at': None,
            'expired_at': None,
            'canceled_at': None,
            'failed_at': None,
            'replaced_at': None,
            'replaced_by': None,
            'replaces': None,
            'asset_id': 'test_asset',
            'symbol': 'SPY',
            'asset_class': 'us_equity',
            'notional': '4250.00',
            'qty': '10',
            'filled_qty': '0',
            'filled_avg_price': None,
            'order_class': 'simple',
            'order_type': 'limit',
            'type': 'limit',
            'side': 'buy',
            'time_in_force': 'day',
            'limit_price': '425.00',
            'stop_price': None,
            'status': 'new',
            'extended_hours': False,
            'trail_percent': None,
            'trail_price': None,
            'hwm': None
        }
        
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_order_data)
        
        order = OptionsOrder(
            symbol='SPY',
            qty=10,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=OrderTimeInForce.DAY,
            limit_price=425.0
        )
        
        with patch.object(alpaca_connector.session, 'post', return_value=mock_response):
            result = await alpaca_connector.place_order(order)
            
            assert result.id == 'test_order_id'
            assert result.symbol == 'SPY'
            assert result.side == 'buy'
            assert result.qty == '10'
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, alpaca_connector):
        """Test order cancellation"""
        mock_response = Mock()
        mock_response.status = 204
        
        with patch.object(alpaca_connector.session, 'delete', return_value=mock_response):
            result = await alpaca_connector.cancel_order('test_order_id')
            assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, alpaca_connector):
        """Test order cancellation failure"""
        mock_response = Mock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value='Order not found')
        
        with patch.object(alpaca_connector.session, 'delete', return_value=mock_response):
            result = await alpaca_connector.cancel_order('test_order_id')
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_pdt_status(self, alpaca_connector):
        """Test PDT status retrieval"""
        mock_account_data = {
            'portfolio_value': '5000.00',
            'day_trade_count': 2,
            'day_trading_buying_power': '10000.00'
        }
        
        with patch.object(alpaca_connector, 'get_account_info', return_value=Mock(**mock_account_data)):
            pdt_status = await alpaca_connector.get_pdt_status()
            
            assert pdt_status['account_value'] == 5000.0
            assert pdt_status['day_trade_count'] == 2
            assert pdt_status['is_pdt'] is True
            assert pdt_status['can_day_trade'] is True
            assert pdt_status['day_trades_remaining'] == 1
    
    @pytest.mark.asyncio
    async def test_validate_order_sufficient_funds(self, alpaca_connector):
        """Test order validation with sufficient funds"""
        mock_account_data = {
            'buying_power': 10000.0,
            'portfolio_value': 5000.0,
            'day_trade_count': 1
        }
        
        order = OptionsOrder(
            symbol='SPY',
            qty=5,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=OrderTimeInForce.DAY,
            limit_price=100.0
        )
        
        with patch.object(alpaca_connector, 'get_account_info', return_value=Mock(**mock_account_data)):
            with patch.object(alpaca_connector, 'get_pdt_status', return_value={'is_pdt': True, 'can_day_trade': True}):
                is_valid, reason = await alpaca_connector.validate_order(order)
                
                assert is_valid is True
                assert reason == "Order validation passed"
    
    @pytest.mark.asyncio
    async def test_validate_order_insufficient_funds(self, alpaca_connector):
        """Test order validation with insufficient funds"""
        mock_account_data = {
            'buying_power': 1000.0,
            'portfolio_value': 5000.0,
            'day_trade_count': 1
        }
        
        order = OptionsOrder(
            symbol='SPY',
            qty=10,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=OrderTimeInForce.DAY,
            limit_price=200.0  # 10 * 200 * 100 = 200,000 > 1,000
        )
        
        with patch.object(alpaca_connector, 'get_account_info', return_value=Mock(**mock_account_data)):
            is_valid, reason = await alpaca_connector.validate_order(order)
            
            assert is_valid is False
            assert "Insufficient buying power" in reason
    
    @pytest.mark.asyncio
    async def test_validate_order_pdt_violation(self, alpaca_connector):
        """Test order validation with PDT violation"""
        mock_account_data = {
            'buying_power': 10000.0,
            'portfolio_value': 5000.0,
            'day_trade_count': 3
        }
        
        order = OptionsOrder(
            symbol='SPY',
            qty=5,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=OrderTimeInForce.DAY,
            limit_price=100.0
        )
        
        with patch.object(alpaca_connector, 'get_account_info', return_value=Mock(**mock_account_data)):
            with patch.object(alpaca_connector, 'get_pdt_status', return_value={'is_pdt': True, 'can_day_trade': False}):
                is_valid, reason = await alpaca_connector.validate_order(order)
                
                assert is_valid is False
                assert "PDT limit reached" in reason
    
    def test_calculate_commission(self, alpaca_connector):
        """Test commission calculation"""
        commission = alpaca_connector.calculate_commission(10)
        expected = (10 * 0.65) + (10 * 0.000119)
        assert commission == expected
    
    @pytest.mark.asyncio
    async def test_get_buying_power_for_options(self, alpaca_connector):
        """Test options buying power calculation"""
        mock_account_data = {
            'day_trading_buying_power': 20000.0,
            'buying_power': 10000.0
        }
        
        with patch.object(alpaca_connector, 'get_account_info', return_value=Mock(**mock_account_data)):
            buying_power = await alpaca_connector.get_buying_power_for_options()
            assert buying_power == 20000.0  # Should use day trading buying power
    
    @pytest.mark.asyncio
    async def test_get_buying_power_for_options_no_day_trading(self, alpaca_connector):
        """Test options buying power when no day trading power"""
        mock_account_data = {
            'day_trading_buying_power': 0.0,
            'buying_power': 10000.0
        }
        
        with patch.object(alpaca_connector, 'get_account_info', return_value=Mock(**mock_account_data)):
            buying_power = await alpaca_connector.get_buying_power_for_options()
            assert buying_power == 10000.0  # Should use regular buying power

class TestOrderFlow:
    """Test complete order flow integration"""
    
    @pytest.fixture
    def mock_connector(self):
        """Create mock connector for order flow testing"""
        connector = Mock(spec=AlpacaConnector)
        connector.session = AsyncMock()
        return connector
    
    @pytest.mark.asyncio
    async def test_complete_order_flow(self, mock_connector):
        """Test complete order flow from validation to execution"""
        # Mock successful validation
        mock_connector.validate_order.return_value = (True, "Order validation passed")
        
        # Mock successful order placement
        mock_order_result = Mock()
        mock_order_result.id = 'test_order_id'
        mock_order_result.status = 'new'
        mock_connector.place_order.return_value = mock_order_result
        
        # Mock order status check
        mock_connector.get_order_status.return_value = mock_order_result
        
        # Create test order
        order = OptionsOrder(
            symbol='SPY',
            qty=5,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=OrderTimeInForce.DAY,
            limit_price=425.0
        )
        
        # Test validation
        is_valid, reason = await mock_connector.validate_order(order)
        assert is_valid is True
        
        # Test order placement
        result = await mock_connector.place_order(order)
        assert result.id == 'test_order_id'
        
        # Test order status check
        status = await mock_connector.get_order_status('test_order_id')
        assert status.status == 'new'

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors"""
        connector = AlpacaConnector()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                await connector.initialize()
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test handling of API errors"""
        connector = AlpacaConnector()
        connector.session = AsyncMock()
        
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='Internal Server Error')
        
        connector.session.get.return_value = mock_response
        
        with pytest.raises(Exception, match="Failed to get account info"):
            await connector.get_account_info()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
