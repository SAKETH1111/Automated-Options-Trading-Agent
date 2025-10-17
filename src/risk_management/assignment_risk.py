"""
Assignment Risk Management and Pin Risk Detection
Advanced risk management for options assignment scenarios

Features:
- Pin risk detection (strikes near spot at expiration)
- Early assignment probability modeling
- Dividend date monitoring
- Cash reserve management
- Auto-close position management
- Assignment notification system
- Risk-adjusted position sizing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import yaml
from loguru import logger
from scipy import stats
import sqlite3
from pathlib import Path

class AssignmentRiskLevel(Enum):
    """Assignment risk classification levels"""
    LOW = "low"           # Deep OTM, low assignment probability
    MEDIUM = "medium"     # Near-the-money, moderate assignment probability
    HIGH = "high"         # Near expiration, high assignment probability
    CRITICAL = "critical" # ITM near expiration, very high assignment probability

class AssignmentTrigger(Enum):
    """Assignment trigger conditions"""
    DIVIDEND_DATE = "dividend_date"
    EXPIRATION_NEAR = "expiration_near"
    DEEP_ITM = "deep_itm"
    PIN_RISK = "pin_risk"
    MANUAL = "manual"

@dataclass
class AssignmentRisk:
    """Assignment risk assessment"""
    position_id: str
    symbol: str
    contract_type: str
    strike: float
    expiration: datetime
    current_price: float
    days_to_expiration: int
    assignment_probability: float
    risk_level: AssignmentRiskLevel
    trigger_reasons: List[AssignmentTrigger]
    cash_requirement: float
    recommended_action: str
    auto_close_threshold: Optional[float] = None
    priority: int = 0

@dataclass
class PinRiskAnalysis:
    """Pin risk analysis for expiration week"""
    symbol: str
    expiration_date: datetime
    current_spot: float
    high_risk_strikes: List[float]
    medium_risk_strikes: List[float]
    risk_zones: Dict[str, List[float]]
    recommended_actions: List[str]
    confidence_level: float

@dataclass
class DividendEvent:
    """Dividend event information"""
    symbol: str
    ex_dividend_date: datetime
    dividend_amount: float
    record_date: datetime
    payment_date: datetime
    assignment_risk_days: List[datetime]

class AssignmentRiskManager:
    """Advanced assignment risk management system"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config = self._load_config(config_path)
        self.risk_config = self.config.get('assignment_risk', {})
        
        # Risk thresholds
        self.pin_risk_threshold = self.risk_config.get('pin_risk_threshold', 2.0)  # $2 from strike
        self.auto_close_days = self.risk_config.get('auto_close_days', 1)  # Auto-close 1 day before expiration
        self.max_assignment_probability = self.risk_config.get('max_assignment_probability', 0.3)  # 30%
        
        # Cash reserve requirements
        self.cash_reserve_multiplier = self.risk_config.get('cash_reserve_multiplier', 1.2)  # 120% of requirement
        self.min_cash_reserve = self.risk_config.get('min_cash_reserve', 1000.0)  # $1000 minimum
        
        # Dividend monitoring
        self.dividend_risk_days = self.risk_config.get('dividend_risk_days', 2)  # 2 days before ex-date
        
        # Risk level thresholds
        self.risk_thresholds = {
            AssignmentRiskLevel.LOW: 0.05,      # < 5% assignment probability
            AssignmentRiskLevel.MEDIUM: 0.20,   # 5-20% assignment probability
            AssignmentRiskLevel.HIGH: 0.50,     # 20-50% assignment probability
            AssignmentRiskLevel.CRITICAL: 1.0   # > 50% assignment probability
        }
        
        # Historical data storage
        self.assignment_history = []
        self.dividend_events = []
        self.pin_risk_data = {}
        
        logger.info("Assignment risk manager initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def calculate_assignment_probability(self, position: Dict[str, Any], 
                                       market_data: Dict[str, Any]) -> float:
        """Calculate assignment probability for a position"""
        try:
            symbol = position['symbol']
            contract_type = position['contract_type']
            strike = position['strike']
            expiration = pd.to_datetime(position['expiration'])
            current_price = market_data.get('current_price', 400.0)
            days_to_expiration = (expiration - datetime.now()).days
            implied_volatility = market_data.get('implied_volatility', 0.2)
            
            # Base probability calculation using Black-Scholes
            base_prob = self._calculate_option_probability(
                current_price, strike, days_to_expiration, implied_volatility, contract_type
            )
            
            # Adjust for dividend risk
            dividend_adjustment = self._calculate_dividend_risk_adjustment(
                symbol, expiration, position['quantity']
            )
            
            # Adjust for pin risk
            pin_adjustment = self._calculate_pin_risk_adjustment(
                current_price, strike, days_to_expiration
            )
            
            # Adjust for time decay
            time_adjustment = self._calculate_time_decay_adjustment(days_to_expiration)
            
            # Calculate final probability
            final_probability = base_prob * dividend_adjustment * pin_adjustment * time_adjustment
            
            return min(max(final_probability, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating assignment probability: {e}")
            return 0.1  # Default 10% probability
    
    def _calculate_option_probability(self, spot: float, strike: float, 
                                    days_to_expiration: int, iv: float, 
                                    contract_type: str) -> float:
        """Calculate option exercise probability using Black-Scholes"""
        try:
            if days_to_expiration <= 0:
                return 1.0 if (contract_type == 'CALL' and spot > strike) or \
                              (contract_type == 'PUT' and spot < strike) else 0.0
            
            # Convert to annualized
            time_to_expiry = days_to_expiration / 365.0
            
            if time_to_expiry <= 0:
                return 0.0
            
            # Calculate d1 and d2
            d1 = (np.log(spot / strike) + (0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
            d2 = d1 - iv * np.sqrt(time_to_expiry)
            
            if contract_type.upper() == 'CALL':
                # Probability that spot > strike at expiration
                probability = 1 - stats.norm.cdf(d2)
            else:  # PUT
                # Probability that spot < strike at expiration
                probability = stats.norm.cdf(-d2)
            
            return probability
            
        except Exception as e:
            logger.error(f"Error calculating option probability: {e}")
            return 0.1  # Default probability
    
    def _calculate_dividend_risk_adjustment(self, symbol: str, expiration: datetime, 
                                          quantity: int) -> float:
        """Calculate dividend risk adjustment factor"""
        try:
            # Check for dividend events near expiration
            dividend_events = self._get_dividend_events(symbol, expiration)
            
            if not dividend_events:
                return 1.0  # No dividend risk
            
            # Calculate adjustment based on dividend amount and timing
            max_adjustment = 1.0
            for dividend in dividend_events:
                days_before_expiration = (expiration - dividend.ex_dividend_date).days
                
                if 0 <= days_before_expiration <= self.dividend_risk_days:
                    # Higher dividend amount and closer to expiration = higher risk
                    dividend_impact = dividend.dividend_amount * quantity * 100
                    risk_multiplier = 1.0 + (dividend_impact / 10000)  # Scale by $10k
                    
                    # Closer to expiration = higher risk
                    time_multiplier = 1.0 + (self.dividend_risk_days - days_before_expiration) * 0.2
                    
                    max_adjustment = max(max_adjustment, risk_multiplier * time_multiplier)
            
            return max_adjustment
            
        except Exception as e:
            logger.error(f"Error calculating dividend risk adjustment: {e}")
            return 1.0
    
    def _calculate_pin_risk_adjustment(self, spot: float, strike: float, 
                                     days_to_expiration: int) -> float:
        """Calculate pin risk adjustment factor"""
        try:
            if days_to_expiration > 5:
                return 1.0  # No pin risk beyond 5 days
            
            # Distance from strike
            distance_from_strike = abs(spot - strike)
            
            # Pin risk increases as expiration approaches and spot gets closer to strike
            if distance_from_strike <= self.pin_risk_threshold:
                pin_risk_factor = 1.0 + (self.pin_risk_threshold - distance_from_strike) / self.pin_risk_threshold
                time_factor = 1.0 + (5 - days_to_expiration) * 0.2
                return pin_risk_factor * time_factor
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating pin risk adjustment: {e}")
            return 1.0
    
    def _calculate_time_decay_adjustment(self, days_to_expiration: int) -> float:
        """Calculate time decay adjustment factor"""
        try:
            if days_to_expiration <= 0:
                return 1.0  # At expiration
            
            # Assignment risk increases exponentially as expiration approaches
            if days_to_expiration <= 1:
                return 2.0  # Double risk on expiration day
            elif days_to_expiration <= 3:
                return 1.5  # 50% higher risk in last 3 days
            elif days_to_expiration <= 7:
                return 1.2  # 20% higher risk in last week
            else:
                return 1.0  # Normal risk beyond a week
            
        except Exception as e:
            logger.error(f"Error calculating time decay adjustment: {e}")
            return 1.0
    
    def assess_assignment_risk(self, positions: List[Dict[str, Any]], 
                             market_data: Dict[str, Any]) -> List[AssignmentRisk]:
        """Assess assignment risk for all positions"""
        try:
            risk_assessments = []
            
            for position in positions:
                # Calculate assignment probability
                assignment_prob = self.calculate_assignment_probability(position, market_data)
                
                # Determine risk level
                risk_level = self._classify_risk_level(assignment_prob)
                
                # Identify trigger reasons
                trigger_reasons = self._identify_trigger_reasons(position, market_data)
                
                # Calculate cash requirement
                cash_requirement = self._calculate_cash_requirement(position)
                
                # Get recommended action
                recommended_action = self._get_recommended_action(
                    position, assignment_prob, risk_level, trigger_reasons
                )
                
                # Set priority
                priority = self._calculate_priority(risk_level, assignment_prob)
                
                # Set auto-close threshold
                auto_close_threshold = self._get_auto_close_threshold(position, risk_level)
                
                risk_assessment = AssignmentRisk(
                    position_id=position['position_id'],
                    symbol=position['symbol'],
                    contract_type=position['contract_type'],
                    strike=position['strike'],
                    expiration=pd.to_datetime(position['expiration']),
                    current_price=market_data.get('current_price', 400.0),
                    days_to_expiration=(pd.to_datetime(position['expiration']) - datetime.now()).days,
                    assignment_probability=assignment_prob,
                    risk_level=risk_level,
                    trigger_reasons=trigger_reasons,
                    cash_requirement=cash_requirement,
                    recommended_action=recommended_action,
                    auto_close_threshold=auto_close_threshold,
                    priority=priority
                )
                
                risk_assessments.append(risk_assessment)
            
            # Sort by priority (highest first)
            risk_assessments.sort(key=lambda x: x.priority, reverse=True)
            
            return risk_assessments
            
        except Exception as e:
            logger.error(f"Error assessing assignment risk: {e}")
            return []
    
    def _classify_risk_level(self, assignment_probability: float) -> AssignmentRiskLevel:
        """Classify assignment risk level"""
        if assignment_probability >= self.risk_thresholds[AssignmentRiskLevel.CRITICAL]:
            return AssignmentRiskLevel.CRITICAL
        elif assignment_probability >= self.risk_thresholds[AssignmentRiskLevel.HIGH]:
            return AssignmentRiskLevel.HIGH
        elif assignment_probability >= self.risk_thresholds[AssignmentRiskLevel.MEDIUM]:
            return AssignmentRiskLevel.MEDIUM
        else:
            return AssignmentRiskLevel.LOW
    
    def _identify_trigger_reasons(self, position: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> List[AssignmentTrigger]:
        """Identify assignment trigger reasons"""
        trigger_reasons = []
        
        symbol = position['symbol']
        expiration = pd.to_datetime(position['expiration'])
        days_to_expiration = (expiration - datetime.now()).days
        
        # Check for dividend risk
        if self._has_dividend_risk(symbol, expiration):
            trigger_reasons.append(AssignmentTrigger.DIVIDEND_DATE)
        
        # Check for expiration risk
        if days_to_expiration <= self.auto_close_days:
            trigger_reasons.append(AssignmentTrigger.EXPIRATION_NEAR)
        
        # Check for deep ITM risk
        current_price = market_data.get('current_price', 400.0)
        strike = position['strike']
        contract_type = position['contract_type']
        
        if ((contract_type == 'CALL' and current_price > strike * 1.1) or
            (contract_type == 'PUT' and current_price < strike * 0.9)):
            trigger_reasons.append(AssignmentTrigger.DEEP_ITM)
        
        # Check for pin risk
        if abs(current_price - strike) <= self.pin_risk_threshold and days_to_expiration <= 5:
            trigger_reasons.append(AssignmentTrigger.PIN_RISK)
        
        return trigger_reasons
    
    def _calculate_cash_requirement(self, position: Dict[str, Any]) -> float:
        """Calculate cash requirement for potential assignment"""
        try:
            contract_type = position['contract_type']
            strike = position['strike']
            quantity = position['quantity']
            
            if contract_type.upper() == 'CALL':
                # Cash-secured call: strike * quantity * 100
                cash_requirement = strike * quantity * 100
            else:  # PUT
                # Put assignment: strike * quantity * 100
                cash_requirement = strike * quantity * 100
            
            # Add buffer
            return cash_requirement * self.cash_reserve_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating cash requirement: {e}")
            return 0.0
    
    def _get_recommended_action(self, position: Dict[str, Any], assignment_prob: float,
                              risk_level: AssignmentRiskLevel, 
                              trigger_reasons: List[AssignmentTrigger]) -> str:
        """Get recommended action based on risk assessment"""
        try:
            if risk_level == AssignmentRiskLevel.CRITICAL:
                return "CLOSE_IMMEDIATELY"
            elif risk_level == AssignmentRiskLevel.HIGH:
                if AssignmentTrigger.EXPIRATION_NEAR in trigger_reasons:
                    return "CLOSE_BEFORE_EXPIRATION"
                else:
                    return "CLOSE_WITHIN_24H"
            elif risk_level == AssignmentRiskLevel.MEDIUM:
                if AssignmentTrigger.DIVIDEND_DATE in trigger_reasons:
                    return "MONITOR_CLOSELY"
                else:
                    return "NORMAL_MONITORING"
            else:
                return "NORMAL_MONITORING"
                
        except Exception as e:
            logger.error(f"Error getting recommended action: {e}")
            return "NORMAL_MONITORING"
    
    def _calculate_priority(self, risk_level: AssignmentRiskLevel, 
                          assignment_prob: float) -> int:
        """Calculate priority score for risk management"""
        priority_scores = {
            AssignmentRiskLevel.CRITICAL: 100,
            AssignmentRiskLevel.HIGH: 75,
            AssignmentRiskLevel.MEDIUM: 50,
            AssignmentRiskLevel.LOW: 25
        }
        
        base_priority = priority_scores[risk_level]
        probability_bonus = assignment_prob * 50  # Up to 50 point bonus
        
        return int(base_priority + probability_bonus)
    
    def _get_auto_close_threshold(self, position: Dict[str, Any], 
                                risk_level: AssignmentRiskLevel) -> Optional[float]:
        """Get auto-close threshold for position"""
        if risk_level in [AssignmentRiskLevel.HIGH, AssignmentRiskLevel.CRITICAL]:
            # Auto-close if assignment probability exceeds threshold
            return self.max_assignment_probability
        
        return None
    
    def _has_dividend_risk(self, symbol: str, expiration: datetime) -> bool:
        """Check if position has dividend risk"""
        dividend_events = self._get_dividend_events(symbol, expiration)
        
        for dividend in dividend_events:
            days_before_expiration = (expiration - dividend.ex_dividend_date).days
            if 0 <= days_before_expiration <= self.dividend_risk_days:
                return True
        
        return False
    
    def _get_dividend_events(self, symbol: str, expiration: datetime) -> List[DividendEvent]:
        """Get dividend events for symbol near expiration"""
        # In production, this would query a dividend database
        # For now, return mock data
        mock_dividends = [
            DividendEvent(
                symbol=symbol,
                ex_dividend_date=expiration - timedelta(days=1),
                dividend_amount=1.50,
                record_date=expiration,
                payment_date=expiration + timedelta(days=30),
                assignment_risk_days=[expiration - timedelta(days=2), expiration - timedelta(days=1)]
            )
        ]
        
        return mock_dividends
    
    def analyze_pin_risk(self, symbol: str, expiration_date: datetime, 
                        current_spot: float, strikes: List[float]) -> PinRiskAnalysis:
        """Analyze pin risk for expiration week"""
        try:
            days_to_expiration = (expiration_date - datetime.now()).days
            
            if days_to_expiration > 7:
                return PinRiskAnalysis(
                    symbol=symbol,
                    expiration_date=expiration_date,
                    current_spot=current_spot,
                    high_risk_strikes=[],
                    medium_risk_strikes=[],
                    risk_zones={},
                    recommended_actions=[],
                    confidence_level=0.0
                )
            
            # Calculate risk for each strike
            high_risk_strikes = []
            medium_risk_strikes = []
            
            for strike in strikes:
                distance = abs(current_spot - strike)
                
                if distance <= self.pin_risk_threshold:
                    high_risk_strikes.append(strike)
                elif distance <= self.pin_risk_threshold * 2:
                    medium_risk_strikes.append(strike)
            
            # Define risk zones
            risk_zones = {
                'high_risk': high_risk_strikes,
                'medium_risk': medium_risk_strikes,
                'low_risk': [s for s in strikes if s not in high_risk_strikes and s not in medium_risk_strikes]
            }
            
            # Generate recommendations
            recommendations = []
            if high_risk_strikes:
                recommendations.append("Close positions at high-risk strikes immediately")
            if medium_risk_strikes:
                recommendations.append("Monitor positions at medium-risk strikes closely")
            
            if days_to_expiration <= 1:
                recommendations.append("Consider closing all near-the-money positions")
            
            # Calculate confidence level
            confidence_level = min(1.0, len(high_risk_strikes) / len(strikes) + 0.5)
            
            return PinRiskAnalysis(
                symbol=symbol,
                expiration_date=expiration_date,
                current_spot=current_spot,
                high_risk_strikes=high_risk_strikes,
                medium_risk_strikes=medium_risk_strikes,
                risk_zones=risk_zones,
                recommended_actions=recommendations,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logger.error(f"Error analyzing pin risk: {e}")
            return PinRiskAnalysis(
                symbol=symbol,
                expiration_date=expiration_date,
                current_spot=current_spot,
                high_risk_strikes=[],
                medium_risk_strikes=[],
                risk_zones={},
                recommended_actions=[],
                confidence_level=0.0
            )
    
    def calculate_cash_reserve_requirement(self, positions: List[Dict[str, Any]]) -> float:
        """Calculate total cash reserve requirement"""
        try:
            total_requirement = 0.0
            
            for position in positions:
                cash_requirement = self._calculate_cash_requirement(position)
                total_requirement += cash_requirement
            
            # Add minimum reserve
            total_requirement = max(total_requirement, self.min_cash_reserve)
            
            return total_requirement
            
        except Exception as e:
            logger.error(f"Error calculating cash reserve requirement: {e}")
            return self.min_cash_reserve
    
    def should_auto_close_position(self, position: Dict[str, Any], 
                                 market_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if position should be auto-closed"""
        try:
            risk_assessments = self.assess_assignment_risk([position], market_data)
            
            if not risk_assessments:
                return False, "No risk assessment available"
            
            risk = risk_assessments[0]
            
            # Auto-close conditions
            if risk.recommended_action in ["CLOSE_IMMEDIATELY", "CLOSE_BEFORE_EXPIRATION"]:
                return True, f"High assignment risk: {risk.assignment_probability:.1%}"
            
            if risk.auto_close_threshold and risk.assignment_probability > risk.auto_close_threshold:
                return True, f"Assignment probability exceeds threshold: {risk.assignment_probability:.1%}"
            
            if risk.days_to_expiration <= 0:
                return True, "Position expired"
            
            return False, "No auto-close conditions met"
            
        except Exception as e:
            logger.error(f"Error determining auto-close: {e}")
            return False, f"Error: {str(e)}"

# Example usage and testing
def main():
    """Test the assignment risk manager"""
    manager = AssignmentRiskManager()
    
    # Test position data
    positions = [
        {
            'position_id': 'POS_001',
            'symbol': 'SPY',
            'contract_type': 'PUT',
            'strike': 400.0,
            'expiration': '2024-01-19',
            'quantity': 10
        },
        {
            'position_id': 'POS_002',
            'symbol': 'QQQ',
            'contract_type': 'CALL',
            'strike': 420.0,
            'expiration': '2024-01-26',
            'quantity': 5
        }
    ]
    
    # Test market data
    market_data = {
        'current_price': 405.0,
        'implied_volatility': 0.22
    }
    
    # Assess assignment risk
    risk_assessments = manager.assess_assignment_risk(positions, market_data)
    
    print("Assignment Risk Assessment:")
    for risk in risk_assessments:
        print(f"\nPosition: {risk.position_id}")
        print(f"Symbol: {risk.symbol} {risk.contract_type} {risk.strike}")
        print(f"Assignment Probability: {risk.assignment_probability:.1%}")
        print(f"Risk Level: {risk.risk_level.value}")
        print(f"Recommended Action: {risk.recommended_action}")
        print(f"Cash Requirement: ${risk.cash_requirement:,.2f}")
    
    # Test pin risk analysis
    pin_analysis = manager.analyze_pin_risk(
        symbol='SPY',
        expiration_date=datetime(2024, 1, 19),
        current_spot=405.0,
        strikes=[395, 400, 405, 410, 415]
    )
    
    print(f"\nPin Risk Analysis:")
    print(f"High Risk Strikes: {pin_analysis.high_risk_strikes}")
    print(f"Medium Risk Strikes: {pin_analysis.medium_risk_strikes}")
    print(f"Recommendations: {pin_analysis.recommended_actions}")
    
    # Test cash reserve requirement
    total_reserve = manager.calculate_cash_reserve_requirement(positions)
    print(f"\nTotal Cash Reserve Required: ${total_reserve:,.2f}")

if __name__ == "__main__":
    main()
