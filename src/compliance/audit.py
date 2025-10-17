"""
Audit Trail & Compliance System for Real Money Trading
Complete audit logging and regulatory compliance for options trading
"""

import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from pathlib import Path
import boto3
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config.settings import get_config
from src.database.session import get_db


class AuditEventType(Enum):
    TRADE_DECISION = "TRADE_DECISION"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    POSITION_CHANGE = "POSITION_CHANGE"
    PARAMETER_CHANGE = "PARAMETER_CHANGE"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    DATA_QUALITY = "DATA_QUALITY"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    RISK_VIOLATION = "RISK_VIOLATION"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    USER_ACTION = "USER_ACTION"


class ComplianceRuleType(Enum):
    PDT_RULE = "PDT_RULE"
    MARGIN_COMPLIANCE = "MARGIN_COMPLIANCE"
    POSITION_LIMITS = "POSITION_LIMITS"
    WASH_SALE = "WASH_SALE"
    TAX_REPORTING = "TAX_REPORTING"
    PATTERN_DAY_TRADING = "PATTERN_DAY_TRADING"


@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    details: Dict[str, Any]
    outcome: str
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    compliance_impact: bool = False
    regulatory_required: bool = False


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    timestamp: datetime
    rule_type: ComplianceRuleType
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    account_id: str
    trade_id: Optional[str]
    position_id: Optional[str]
    violation_data: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class TaxReportRecord:
    """Tax reporting record for Form 8949"""
    trade_id: str
    symbol: str
    option_symbol: str
    date_acquired: datetime
    date_sold: datetime
    proceeds: float
    cost_basis: float
    gain_loss: float
    wash_sale_adjustment: float = 0.0
    reported: bool = False


class AuditTrailSystem:
    """
    Comprehensive audit trail system for real money trading
    
    Features:
    - Complete event logging
    - Regulatory compliance monitoring
    - Tax reporting preparation
    - Risk violation tracking
    - Data retention management
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or get_config()
        self.db = get_db()
        
        # Audit configuration
        self.audit_config = {
            'retention_days': 2555,  # 7 years
            'log_rotation_days': 30,
            'backup_to_s3': True,
            'encrypt_logs': True,
            'real_time_alerts': True
        }
        
        # Compliance rules
        self.compliance_rules = self._initialize_compliance_rules()
        
        # Tax tracking
        self.tax_records = []
        self.wash_sale_tracker = {}
        
        # File paths
        self.log_dir = Path("logs/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # S3 client for backup
        self.s3_client = None
        if self.audit_config['backup_to_s3']:
            self._initialize_s3()
        
        logger.info("AuditTrailSystem initialized")
    
    def _initialize_compliance_rules(self) -> Dict[str, Dict]:
        """Initialize compliance rules"""
        return {
            'pdt_rule': {
                'enabled': True,
                'min_account_balance': 25000,
                'max_day_trades': 3,
                'description': 'Pattern Day Trading rule'
            },
            'margin_compliance': {
                'enabled': True,
                'max_leverage': 4.0,
                'description': 'Margin requirement compliance'
            },
            'position_limits': {
                'enabled': True,
                'max_positions_per_symbol': 10,
                'max_portfolio_heat': 50.0,
                'description': 'Position size limits'
            },
            'wash_sale': {
                'enabled': True,
                'wash_sale_period_days': 61,  # 30 days before + after + day of
                'description': 'Wash sale rule (IRS)'
            },
            'tax_reporting': {
                'enabled': True,
                'report_all_trades': True,
                'form_8949_required': True,
                'description': 'Tax reporting requirements'
            }
        }
    
    def _initialize_s3(self):
        """Initialize S3 client for log backup"""
        try:
            self.s3_client = boto3.client('s3')
            logger.info("S3 client initialized for audit log backup")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    # Core Audit Logging
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        details: Dict[str, Any],
        outcome: str,
        user_id: str = None,
        session_id: str = None,
        risk_level: str = "LOW"
    ) -> str:
        """
        Log an audit event
        
        Returns:
            event_id: Unique identifier for the logged event
        """
        try:
            event_id = str(uuid.uuid4())
            
            # Determine compliance impact
            compliance_impact = self._check_compliance_impact(event_type, details)
            regulatory_required = self._is_regulatory_required(event_type)
            
            # Create audit event
            event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                action=action,
                details=details,
                outcome=outcome,
                risk_level=risk_level,
                compliance_impact=compliance_impact,
                regulatory_required=regulatory_required
            )
            
            # Store in database
            self._store_audit_event(event)
            
            # Write to log file
            self._write_to_log_file(event)
            
            # Check for compliance violations
            if compliance_impact:
                self._check_compliance_violations(event)
            
            # Send real-time alert if high risk
            if risk_level in ["HIGH", "CRITICAL"]:
                self._send_audit_alert(event)
            
            logger.info(f"Audit event logged: {event_id} - {action}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            return None
    
    def log_trade_decision(
        self,
        trade_params: Dict[str, Any],
        decision: str,
        reasoning: str,
        risk_assessment: Dict[str, Any]
    ) -> str:
        """Log trade decision process"""
        details = {
            'trade_params': trade_params,
            'decision': decision,
            'reasoning': reasoning,
            'risk_assessment': risk_assessment,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        risk_level = "HIGH" if decision == "REJECT" and risk_assessment.get('critical_risk') else "MEDIUM"
        
        return self.log_event(
            event_type=AuditEventType.TRADE_DECISION,
            action=f"Trade decision: {decision}",
            details=details,
            outcome=decision,
            risk_level=risk_level
        )
    
    def log_trade_execution(
        self,
        trade_id: str,
        execution_details: Dict[str, Any],
        fill_prices: List[float],
        slippage: float,
        commission: float
    ) -> str:
        """Log trade execution"""
        details = {
            'trade_id': trade_id,
            'execution_details': execution_details,
            'fill_prices': fill_prices,
            'slippage': slippage,
            'commission': commission,
            'execution_time': datetime.utcnow().isoformat()
        }
        
        # Determine risk level based on slippage
        risk_level = "HIGH" if slippage > 0.1 else "MEDIUM" if slippage > 0.05 else "LOW"
        
        return self.log_event(
            event_type=AuditEventType.TRADE_EXECUTION,
            action=f"Trade executed: {trade_id}",
            details=details,
            outcome="EXECUTED",
            risk_level=risk_level
        )
    
    def log_position_change(
        self,
        position_id: str,
        change_type: str,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        reason: str
    ) -> str:
        """Log position state changes"""
        details = {
            'position_id': position_id,
            'change_type': change_type,
            'old_state': old_state,
            'new_state': new_state,
            'reason': reason,
            'change_time': datetime.utcnow().isoformat()
        }
        
        return self.log_event(
            event_type=AuditEventType.POSITION_CHANGE,
            action=f"Position {change_type}: {position_id}",
            details=details,
            outcome="CHANGED"
        )
    
    def log_parameter_change(
        self,
        parameter_name: str,
        old_value: Any,
        new_value: Any,
        change_reason: str,
        approved_by: str = None
    ) -> str:
        """Log parameter changes"""
        details = {
            'parameter_name': parameter_name,
            'old_value': old_value,
            'new_value': new_value,
            'change_reason': change_reason,
            'approved_by': approved_by,
            'change_time': datetime.utcnow().isoformat()
        }
        
        # High risk for critical parameters
        critical_params = ['max_daily_loss', 'max_position_size', 'trading_mode']
        risk_level = "HIGH" if parameter_name in critical_params else "MEDIUM"
        
        return self.log_event(
            event_type=AuditEventType.PARAMETER_CHANGE,
            action=f"Parameter changed: {parameter_name}",
            details=details,
            outcome="CHANGED",
            risk_level=risk_level
        )
    
    def log_circuit_breaker(
        self,
        breaker_level: str,
        trigger_reason: str,
        action_taken: str,
        affected_positions: List[str]
    ) -> str:
        """Log circuit breaker events"""
        details = {
            'breaker_level': breaker_level,
            'trigger_reason': trigger_reason,
            'action_taken': action_taken,
            'affected_positions': affected_positions,
            'trigger_time': datetime.utcnow().isoformat()
        }
        
        return self.log_event(
            event_type=AuditEventType.CIRCUIT_BREAKER,
            action=f"Circuit breaker: {breaker_level}",
            details=details,
            outcome=action_taken,
            risk_level="CRITICAL"
        )
    
    # Compliance Monitoring
    
    def check_pdt_compliance(self, account_balance: float, day_trades_count: int) -> Dict[str, Any]:
        """Check Pattern Day Trading compliance"""
        try:
            rule = self.compliance_rules['pdt_rule']
            
            if not rule['enabled']:
                return {'compliant': True, 'reason': 'PDT rule disabled'}
            
            # Check account balance
            if account_balance < rule['min_account_balance']:
                # Check day trade count
                if day_trades_count > rule['max_day_trades']:
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        timestamp=datetime.utcnow(),
                        rule_type=ComplianceRuleType.PATTERN_DAY_TRADING,
                        severity="CRITICAL",
                        description=f"PDT violation: {day_trades_count} day trades with balance ${account_balance:,.0f}",
                        account_id="main",
                        violation_data={
                            'account_balance': account_balance,
                            'day_trades_count': day_trades_count,
                            'min_balance_required': rule['min_account_balance'],
                            'max_day_trades_allowed': rule['max_day_trades']
                        }
                    )
                    
                    self._record_compliance_violation(violation)
                    
                    return {
                        'compliant': False,
                        'violation': violation,
                        'reason': 'Account below $25K and exceeded day trade limit'
                    }
            
            return {
                'compliant': True,
                'reason': 'PDT requirements met',
                'account_balance': account_balance,
                'day_trades': day_trades_count
            }
            
        except Exception as e:
            logger.error(f"Error checking PDT compliance: {e}")
            return {'compliant': False, 'error': str(e)}
    
    def check_wash_sale_rule(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Check wash sale rule compliance"""
        try:
            rule = self.compliance_rules['wash_sale']
            
            if not rule['enabled']:
                return {'wash_sale': False}
            
            symbol = trade.get('symbol', '')
            trade_date = trade.get('trade_date', datetime.utcnow())
            side = trade.get('side', '')
            quantity = trade.get('quantity', 0)
            
            # Check for wash sale in the period
            wash_sale_period_start = trade_date - timedelta(days=30)
            wash_sale_period_end = trade_date + timedelta(days=30)
            
            # Get trades in wash sale period
            with self.db.get_session() as session:
                # This would query actual trade history
                # Simplified for example
                potential_wash_sales = []
            
            if potential_wash_sales:
                violation = ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    rule_type=ComplianceRuleType.WASH_SALE,
                    severity="MEDIUM",
                    description=f"Potential wash sale detected for {symbol}",
                    account_id="main",
                    trade_id=trade.get('trade_id'),
                    violation_data={
                        'symbol': symbol,
                        'trade_date': trade_date.isoformat(),
                        'potential_wash_sales': potential_wash_sales
                    }
                )
                
                self._record_compliance_violation(violation)
                
                return {
                    'wash_sale': True,
                    'violation': violation,
                    'adjustment_required': True
                }
            
            return {'wash_sale': False}
            
        except Exception as e:
            logger.error(f"Error checking wash sale rule: {e}")
            return {'wash_sale': False, 'error': str(e)}
    
    def check_margin_compliance(self, positions: List[Dict], account_balance: float) -> Dict[str, Any]:
        """Check margin compliance"""
        try:
            rule = self.compliance_rules['margin_compliance']
            
            if not rule['enabled']:
                return {'compliant': True}
            
            # Calculate total margin requirement
            total_margin_required = 0.0
            for position in positions:
                margin_req = position.get('margin_requirement', 0)
                quantity = position.get('quantity', 1)
                total_margin_required += margin_req * quantity
            
            # Check leverage
            leverage = total_margin_required / account_balance if account_balance > 0 else 0
            
            if leverage > rule['max_leverage']:
                violation = ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow(),
                    rule_type=ComplianceRuleType.MARGIN_COMPLIANCE,
                    severity="HIGH",
                    description=f"Margin violation: leverage {leverage:.2f}x exceeds limit {rule['max_leverage']}x",
                    account_id="main",
                    violation_data={
                        'total_margin_required': total_margin_required,
                        'account_balance': account_balance,
                        'leverage': leverage,
                        'max_leverage': rule['max_leverage']
                    }
                )
                
                self._record_compliance_violation(violation)
                
                return {
                    'compliant': False,
                    'violation': violation,
                    'leverage': leverage,
                    'margin_required': total_margin_required
                }
            
            return {
                'compliant': True,
                'leverage': leverage,
                'margin_required': total_margin_required
            }
            
        except Exception as e:
            logger.error(f"Error checking margin compliance: {e}")
            return {'compliant': False, 'error': str(e)}
    
    # Tax Reporting
    
    def create_tax_record(self, trade: Dict[str, Any]) -> TaxReportRecord:
        """Create tax reporting record"""
        try:
            return TaxReportRecord(
                trade_id=trade.get('trade_id', str(uuid.uuid4())),
                symbol=trade.get('symbol', ''),
                option_symbol=trade.get('option_symbol', ''),
                date_acquired=trade.get('entry_date', datetime.utcnow()),
                date_sold=trade.get('exit_date', datetime.utcnow()),
                proceeds=trade.get('exit_price', 0) * trade.get('quantity', 0),
                cost_basis=trade.get('entry_price', 0) * trade.get('quantity', 0),
                gain_loss=trade.get('pnl', 0)
            )
            
        except Exception as e:
            logger.error(f"Error creating tax record: {e}")
            return None
    
    def generate_form_8949_data(self, year: int) -> pd.DataFrame:
        """Generate Form 8949 data for tax reporting"""
        try:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            # Get all closed trades for the year
            with self.db.get_session() as session:
                # This would query actual trade data
                # Simplified for example
                trades = []
            
            # Convert to Form 8949 format
            form_data = []
            for trade in trades:
                if trade.get('status') == 'closed' and trade.get('pnl') != 0:
                    form_data.append({
                        'Description': f"{trade.get('quantity', 0)} {trade.get('option_symbol', '')}",
                        'Date Acquired': trade.get('entry_date', '').strftime('%m/%d/%Y'),
                        'Date Sold': trade.get('exit_date', '').strftime('%m/%d/%Y'),
                        'Proceeds': trade.get('exit_price', 0) * trade.get('quantity', 0),
                        'Cost Basis': trade.get('entry_price', 0) * trade.get('quantity', 0),
                        'Gain/Loss': trade.get('pnl', 0),
                        'Wash Sale': 'Yes' if trade.get('wash_sale', False) else 'No'
                    })
            
            return pd.DataFrame(form_data)
            
        except Exception as e:
            logger.error(f"Error generating Form 8949 data: {e}")
            return pd.DataFrame()
    
    # Data Management
    
    def _store_audit_event(self, event: AuditEvent):
        """Store audit event in database"""
        try:
            with self.db.get_session() as session:
                # In real implementation, would store in audit_events table
                # For now, just log
                logger.debug(f"Storing audit event: {event.event_id}")
                
        except Exception as e:
            logger.error(f"Error storing audit event: {e}")
    
    def _write_to_log_file(self, event: AuditEvent):
        """Write audit event to log file"""
        try:
            # Create daily log file
            log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            
            # Convert event to JSON
            event_dict = asdict(event)
            event_dict['timestamp'] = event.timestamp.isoformat()
            event_dict['event_type'] = event.event_type.value
            
            # Write to file
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
            
            # Backup to S3 if enabled
            if self.audit_config['backup_to_s3'] and self.s3_client:
                self._backup_to_s3(log_file)
                
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    
    def _backup_to_s3(self, log_file: Path):
        """Backup log file to S3"""
        try:
            if not self.s3_client:
                return
            
            s3_key = f"audit-logs/{log_file.name}"
            self.s3_client.upload_file(str(log_file), 'audit-backups', s3_key)
            logger.debug(f"Backed up {log_file.name} to S3")
            
        except Exception as e:
            logger.error(f"Error backing up to S3: {e}")
    
    def _record_compliance_violation(self, violation: ComplianceViolation):
        """Record compliance violation"""
        try:
            # Store in database
            with self.db.get_session() as session:
                # In real implementation, would store in compliance_violations table
                logger.warning(f"Compliance violation: {violation.violation_id}")
            
            # Send alert
            self._send_compliance_alert(violation)
            
            # Log the violation
            self.log_event(
                event_type=AuditEventType.COMPLIANCE_CHECK,
                action=f"Compliance violation: {violation.rule_type.value}",
                details=asdict(violation),
                outcome="VIOLATION",
                risk_level=violation.severity
            )
            
        except Exception as e:
            logger.error(f"Error recording compliance violation: {e}")
    
    def _check_compliance_impact(self, event_type: AuditEventType, details: Dict[str, Any]) -> bool:
        """Check if event has compliance impact"""
        compliance_events = [
            AuditEventType.TRADE_EXECUTION,
            AuditEventType.POSITION_CHANGE,
            AuditEventType.CIRCUIT_BREAKER,
            AuditEventType.RISK_VIOLATION
        ]
        
        return event_type in compliance_events
    
    def _is_regulatory_required(self, event_type: AuditEventType) -> bool:
        """Check if event is required for regulatory reporting"""
        regulatory_events = [
            AuditEventType.TRADE_EXECUTION,
            AuditEventType.POSITION_CHANGE,
            AuditEventType.RISK_VIOLATION,
            AuditEventType.COMPLIANCE_CHECK
        ]
        
        return event_type in regulatory_events
    
    def _check_compliance_violations(self, event: AuditEvent):
        """Check for compliance violations based on event"""
        try:
            # Check PDT rule
            if event.event_type == AuditEventType.TRADE_EXECUTION:
                # Check day trade count and account balance
                # Implementation would check actual values
                pass
            
            # Check margin compliance
            if event.event_type == AuditEventType.POSITION_CHANGE:
                # Check margin requirements
                # Implementation would check actual positions
                pass
                
        except Exception as e:
            logger.error(f"Error checking compliance violations: {e}")
    
    def _send_audit_alert(self, event: AuditEvent):
        """Send audit alert for high-risk events"""
        try:
            # In real implementation, would send email/SMS/Slack alert
            logger.warning(f"AUDIT ALERT: {event.action} - Risk: {event.risk_level}")
            
        except Exception as e:
            logger.error(f"Error sending audit alert: {e}")
    
    def _send_compliance_alert(self, violation: ComplianceViolation):
        """Send compliance violation alert"""
        try:
            # In real implementation, would send critical alert
            logger.critical(f"COMPLIANCE VIOLATION: {violation.rule_type.value} - {violation.description}")
            
        except Exception as e:
            logger.error(f"Error sending compliance alert: {e}")
    
    # Reporting and Analytics
    
    def generate_audit_report(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        try:
            # Read audit logs for period
            report_data = {
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'summary': {
                    'total_events': 0,
                    'high_risk_events': 0,
                    'compliance_violations': 0,
                    'trade_executions': 0
                },
                'events_by_type': {},
                'risk_distribution': {},
                'compliance_status': {}
            }
            
            # Process daily log files
            current_date = start_date.date()
            while current_date <= end_date.date():
                log_file = self.log_dir / f"audit_{current_date.strftime('%Y%m%d')}.jsonl"
                
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                event_data = json.loads(line.strip())
                                event_date = datetime.fromisoformat(event_data['timestamp']).date()
                                
                                if start_date.date() <= event_date <= end_date.date():
                                    report_data['summary']['total_events'] += 1
                                    
                                    # Count by type
                                    event_type = event_data['event_type']
                                    report_data['events_by_type'][event_type] = \
                                        report_data['events_by_type'].get(event_type, 0) + 1
                                    
                                    # Count high risk events
                                    if event_data['risk_level'] in ['HIGH', 'CRITICAL']:
                                        report_data['summary']['high_risk_events'] += 1
                                    
                                    # Count trade executions
                                    if event_type == 'TRADE_EXECUTION':
                                        report_data['summary']['trade_executions'] += 1
                                    
                                    # Risk distribution
                                    risk_level = event_data['risk_level']
                                    report_data['risk_distribution'][risk_level] = \
                                        report_data['risk_distribution'].get(risk_level, 0) + 1
                                        
                            except Exception as e:
                                logger.error(f"Error processing log line: {e}")
                
                current_date += timedelta(days=1)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Error generating audit report: {e}")
            return {'error': str(e)}
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get current compliance status summary"""
        try:
            summary = {
                'rules_enabled': {},
                'recent_violations': [],
                'compliance_score': 100,
                'last_check': datetime.utcnow().isoformat()
            }
            
            # Check which rules are enabled
            for rule_name, rule_config in self.compliance_rules.items():
                summary['rules_enabled'][rule_name] = rule_config['enabled']
            
            # Get recent violations (last 30 days)
            # In real implementation, would query database
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting compliance summary: {e}")
            return {'error': str(e)}
    
    def cleanup_old_logs(self):
        """Clean up old audit logs based on retention policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.audit_config['retention_days'])
            
            for log_file in self.log_dir.glob("audit_*.jsonl"):
                # Parse date from filename
                try:
                    date_str = log_file.stem.replace('audit_', '')
                    file_date = datetime.strptime(date_str, '%Y%m%d').date()
                    
                    if file_date < cutoff_date.date():
                        log_file.unlink()
                        logger.info(f"Deleted old audit log: {log_file.name}")
                        
                except ValueError:
                    logger.warning(f"Could not parse date from filename: {log_file.name}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")


# Example usage
if __name__ == "__main__":
    # Test the audit system
    audit_system = AuditTrailSystem()
    
    # Log a trade decision
    trade_params = {
        'strategy': 'bull_put_spread',
        'symbol': 'SPY',
        'max_loss': 500
    }
    
    event_id = audit_system.log_trade_decision(
        trade_params=trade_params,
        decision="APPROVE",
        reasoning="Passed all validation checks",
        risk_assessment={'risk_score': 0.3, 'critical_risk': False}
    )
    
    print(f"Logged trade decision: {event_id}")
    
    # Check PDT compliance
    pdt_check = audit_system.check_pdt_compliance(
        account_balance=30000,
        day_trades_count=2
    )
    
    print(f"PDT compliance: {pdt_check}")
    
    # Generate audit report
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    report = audit_system.generate_audit_report(start_date, end_date)
    print(f"Audit report: {report}")
    
    # Get compliance summary
    compliance = audit_system.get_compliance_summary()
    print(f"Compliance summary: {compliance}")
