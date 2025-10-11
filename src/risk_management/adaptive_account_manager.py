"""
Adaptive Account Manager
Automatically adjusts trading parameters based on account size
"""

from typing import Dict, List, Tuple
from datetime import datetime
from loguru import logger


class AccountTier:
    """Account tier definition"""
    
    def __init__(
        self,
        name: str,
        min_balance: float,
        max_balance: float,
        risk_pct: float,
        max_positions: int,
        symbols: List[str],
        spread_widths: List[int],
        dte_range: List[int]
    ):
        self.name = name
        self.min_balance = min_balance
        self.max_balance = max_balance
        self.risk_pct = risk_pct
        self.max_positions = max_positions
        self.symbols = symbols
        self.spread_widths = spread_widths
        self.dte_range = dte_range


class AdaptiveAccountManager:
    """
    Manages trading parameters based on account size
    Automatically scales risk, symbols, and strategies
    """
    
    def __init__(self):
        # Define account tiers
        self.tiers = {
            'micro': AccountTier(
                name='Micro',
                min_balance=1000,
                max_balance=2500,
                risk_pct=12.0,
                max_positions=1,
                symbols=['SQQQ', 'UVXY', 'TZA', 'GDX'],
                spread_widths=[1, 2],
                dte_range=[7, 14]
            ),
            'small': AccountTier(
                name='Small',
                min_balance=2500,
                max_balance=5000,
                risk_pct=8.0,
                max_positions=2,
                symbols=['GDX', 'XLF', 'TLT', 'SQQQ'],
                spread_widths=[2, 3, 5],
                dte_range=[14, 30]
            ),
            'medium': AccountTier(
                name='Medium',
                min_balance=5000,
                max_balance=10000,
                risk_pct=5.0,
                max_positions=3,
                symbols=['XLF', 'TLT', 'IWM', 'SPY'],
                spread_widths=[3, 5],
                dte_range=[21, 45]
            ),
            'standard': AccountTier(
                name='Standard',
                min_balance=10000,
                max_balance=25000,
                risk_pct=3.0,
                max_positions=4,
                symbols=['SPY', 'QQQ', 'IWM', 'DIA'],
                spread_widths=[5, 10],
                dte_range=[30, 45]
            ),
            'large': AccountTier(
                name='Large',
                min_balance=25000,
                max_balance=float('inf'),
                risk_pct=2.0,
                max_positions=6,
                symbols=['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE'],
                spread_widths=[5, 10, 15],
                dte_range=[30, 60]
            )
        }
        
        logger.info("Adaptive Account Manager initialized with 5 tiers")
    
    def get_account_tier(self, account_balance: float) -> AccountTier:
        """
        Determine account tier based on balance
        
        Args:
            account_balance: Current account balance
            
        Returns:
            AccountTier object
        """
        for tier_name, tier in self.tiers.items():
            if tier.min_balance <= account_balance < tier.max_balance:
                logger.info(f"Account ${account_balance:,.0f} classified as: {tier.name}")
                return tier
        
        # Default to micro if below minimum
        return self.tiers['micro']
    
    def get_trading_parameters(
        self,
        account_balance: float,
        iv_rank: float = 50,
        ml_confidence: float = 0.70,
        recent_win_rate: float = 0.70
    ) -> Dict:
        """
        Get adaptive trading parameters
        
        Args:
            account_balance: Current account balance
            iv_rank: Current IV rank (0-100)
            ml_confidence: ML model confidence (0-1)
            recent_win_rate: Recent win rate (0-1)
            
        Returns:
            Dictionary of trading parameters
        """
        tier = self.get_account_tier(account_balance)
        
        # Base parameters from tier
        base_risk_pct = tier.risk_pct
        
        # Adjust risk based on conditions
        risk_adjustments = {
            'base': base_risk_pct,
            'iv_adjustment': 1.0,
            'confidence_adjustment': 1.0,
            'performance_adjustment': 1.0
        }
        
        # IV-based adjustment
        if iv_rank > 50:
            risk_adjustments['iv_adjustment'] = 1.2  # High IV = more premium
        elif iv_rank < 30:
            risk_adjustments['iv_adjustment'] = 0.7  # Low IV = be selective
        
        # ML Confidence adjustment
        if ml_confidence > 0.80:
            risk_adjustments['confidence_adjustment'] = 1.3
        elif ml_confidence < 0.60:
            risk_adjustments['confidence_adjustment'] = 0.7
        
        # Performance adjustment
        if recent_win_rate > 0.75:
            risk_adjustments['performance_adjustment'] = 1.1
        elif recent_win_rate < 0.60:
            risk_adjustments['performance_adjustment'] = 0.6
        
        # Calculate final risk
        final_risk_pct = (
            base_risk_pct *
            risk_adjustments['iv_adjustment'] *
            risk_adjustments['confidence_adjustment'] *
            risk_adjustments['performance_adjustment']
        )
        
        # Clamp risk
        min_risk = 0.5
        max_risk = tier.risk_pct * 1.5  # Can't go more than 1.5x base
        final_risk_pct = max(min_risk, min(max_risk, final_risk_pct))
        
        # Determine DTE based on multiple factors
        dte_range = self._select_optimal_dte(
            tier=tier,
            iv_rank=iv_rank,
            ml_confidence=ml_confidence,
            account_balance=account_balance
        )
        
        return {
            'tier': tier.name,
            'tier_name': tier.name,
            'account_balance': account_balance,
            
            # Risk parameters
            'risk_per_trade_pct': round(final_risk_pct, 2),
            'risk_per_trade_dollars': round(account_balance * final_risk_pct / 100, 2),
            'max_positions': tier.max_positions,
            'max_portfolio_heat_pct': tier.risk_pct * tier.max_positions,
            
            # Trading parameters
            'symbols': tier.symbols,
            'spread_widths': tier.spread_widths,
            'dte_range': dte_range,
            
            # Adjustments applied
            'adjustments': risk_adjustments,
            'final_risk_multiplier': round(
                risk_adjustments['iv_adjustment'] *
                risk_adjustments['confidence_adjustment'] *
                risk_adjustments['performance_adjustment'],
                2
            )
        }
    
    def _select_optimal_dte(
        self,
        tier: AccountTier,
        iv_rank: float,
        ml_confidence: float,
        account_balance: float
    ) -> List[int]:
        """
        Intelligently select DTE range based on conditions
        
        Returns:
            [min_dte, max_dte]
        """
        # Start with tier default
        dte_range = tier.dte_range.copy()
        
        # Small accounts prefer weekly for faster income
        if account_balance < 2500:
            dte_range = [7, 14]
        
        # High IV + small account = definitely weekly
        if iv_rank > 50 and account_balance < 5000:
            dte_range = [7, 14]
            logger.info("High IV + small account → Using weekly options")
        
        # Low IV = need monthly for decent credit
        elif iv_rank < 30:
            dte_range = [30, 45]
            logger.info("Low IV → Using monthly options for better credit")
        
        # High ML confidence = can use shorter DTE
        elif ml_confidence > 0.80:
            dte_range = [7, 21]  # Flexible, leaning short
            logger.info("High ML confidence → Using shorter DTE")
        
        # Low ML confidence = need more time
        elif ml_confidence < 0.60:
            dte_range = [30, 45]
            logger.info("Low ML confidence → Using longer DTE for safety")
        
        return dte_range
    
    def calculate_position_size(
        self,
        account_balance: float,
        max_loss_per_contract: float,
        ml_confidence: float = 0.70
    ) -> Dict:
        """
        Calculate position size based on account tier
        
        Args:
            account_balance: Current account balance
            max_loss_per_contract: Maximum loss per contract
            ml_confidence: ML model confidence
            
        Returns:
            Position sizing details
        """
        # Get tier parameters
        params = self.get_trading_parameters(
            account_balance=account_balance,
            ml_confidence=ml_confidence
        )
        
        # Calculate position size
        risk_dollars = params['risk_per_trade_dollars']
        quantity = int(risk_dollars / max_loss_per_contract) if max_loss_per_contract > 0 else 1
        quantity = max(1, quantity)  # At least 1 contract
        
        # Check if position is too large for account tier
        max_positions = params['max_positions']
        if quantity > max_positions:
            logger.warning(f"Position size {quantity} exceeds tier max {max_positions}, capping")
            quantity = max_positions
        
        # Calculate actual risk
        actual_risk_dollars = quantity * max_loss_per_contract
        actual_risk_pct = (actual_risk_dollars / account_balance) * 100
        
        return {
            'quantity': quantity,
            'risk_dollars': actual_risk_dollars,
            'risk_pct': round(actual_risk_pct, 2),
            'tier': params['tier'],
            'max_loss_per_contract': max_loss_per_contract,
            'max_loss_total': actual_risk_dollars,
            'can_trade': actual_risk_dollars <= (account_balance * 0.5)  # Safety check
        }
    
    def should_trade(
        self,
        account_balance: float,
        current_positions: int,
        current_drawdown_pct: float,
        ml_confidence: float
    ) -> Tuple[bool, str]:
        """
        Determine if account should trade based on current state
        
        Returns:
            (can_trade: bool, reason: str)
        """
        tier = self.get_account_tier(account_balance)
        
        # Check 1: Max positions
        if current_positions >= tier.max_positions:
            return False, f"Max positions reached ({tier.max_positions})"
        
        # Check 2: Drawdown limit
        max_dd = 20 if tier.name == 'Micro' else 15
        if current_drawdown_pct > max_dd:
            return False, f"Drawdown too high ({current_drawdown_pct:.1f}% > {max_dd}%)"
        
        # Check 3: ML Confidence
        min_confidence = 0.55 if tier.name == 'Micro' else 0.65
        if ml_confidence < min_confidence:
            return False, f"ML confidence too low ({ml_confidence:.2f} < {min_confidence})"
        
        # Check 4: Minimum balance
        if account_balance < tier.min_balance:
            return False, f"Account balance too low (${account_balance:,.0f} < ${tier.min_balance:,.0f})"
        
        return True, "All checks passed"
    
    def get_recommended_symbols(
        self,
        account_balance: float,
        iv_ranks: Dict[str, float] = None
    ) -> List[str]:
        """
        Get recommended symbols based on account tier and IV environment
        
        Args:
            account_balance: Current account balance
            iv_ranks: Dictionary of symbol: iv_rank
            
        Returns:
            List of recommended symbols (sorted by opportunity)
        """
        tier = self.get_account_tier(account_balance)
        symbols = tier.symbols.copy()
        
        if iv_ranks:
            # Sort by IV rank (higher = better for selling premium)
            symbols_with_iv = [(sym, iv_ranks.get(sym, 50)) for sym in symbols]
            symbols_with_iv.sort(key=lambda x: x[1], reverse=True)
            symbols = [sym for sym, _ in symbols_with_iv]
        
        logger.info(f"Recommended symbols for {tier.name} tier: {symbols}")
        return symbols
    
    def format_tier_summary(self, account_balance: float) -> str:
        """
        Get human-readable summary of current tier
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Formatted string
        """
        tier = self.get_account_tier(account_balance)
        
        summary = f"""
📊 Account Tier: {tier.name}
💰 Balance: ${account_balance:,.2f}
📈 Risk per Trade: {tier.risk_pct}% (${account_balance * tier.risk_pct / 100:,.2f})
🎯 Max Positions: {tier.max_positions}
📋 Symbols: {', '.join(tier.symbols)}
📏 Spread Widths: ${', $'.join(map(str, tier.spread_widths))}
📅 DTE Range: {tier.dte_range[0]}-{tier.dte_range[1]} days
"""
        
        return summary.strip()


# Example usage
if __name__ == "__main__":
    manager = AdaptiveAccountManager()
    
    # Test different account sizes
    test_balances = [1500, 3500, 7500, 15000, 50000]
    
    for balance in test_balances:
        print(f"\n{'='*60}")
        print(f"Testing ${balance:,} account:")
        print(manager.format_tier_summary(balance))
        
        # Get trading parameters
        params = manager.get_trading_parameters(
            account_balance=balance,
            iv_rank=45,
            ml_confidence=0.72
        )
        
        print(f"\n💡 Trading Parameters:")
        print(f"   Risk: {params['risk_per_trade_pct']}% = ${params['risk_per_trade_dollars']:,.2f}")
        print(f"   DTE: {params['dte_range']}")
        print(f"   Symbols: {', '.join(params['symbols'][:3])}")
        
        # Test position sizing
        example_risks = [90, 180, 450, 900]
        for risk in example_risks:
            if risk <= params['risk_per_trade_dollars'] * 1.5:
                size = manager.calculate_position_size(balance, risk, 0.72)
                if size['can_trade']:
                    print(f"   Can trade: {size['quantity']} contract(s) @ ${risk} risk")
                    break

