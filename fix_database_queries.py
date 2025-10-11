#!/usr/bin/env python3
"""
Script to fix database query issues across all files
"""

import os
import re
from pathlib import Path

def fix_database_queries():
    """Fix all database query issues in the codebase"""
    
    # Files that need fixing
    files_to_fix = [
        "src/automation/performance_tracker.py",
        "src/risk_management/circuit_breaker.py", 
        "src/risk_management/portfolio_risk.py",
        "src/ml/feature_engineer.py",
        "src/backtesting/strategy_tester.py",
        "src/options/unusual_activity.py",
        "src/options/opportunity_finder.py",
        "src/options/iv_tracker.py",
        "src/analysis/analyzer.py"
    ]
    
    # Pattern to find direct .query() calls on self.db
    pattern = r'self\.db\.query\('
    replacement = 'with self.db.get_session() as session:\n                session.query('
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"Fixing {file_path}...")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Fix the pattern
            if re.search(pattern, content):
                # More sophisticated replacement needed
                lines = content.split('\n')
                new_lines = []
                
                for i, line in enumerate(lines):
                    if 'self.db.query(' in line:
                        # Find the indentation
                        indent = len(line) - len(line.lstrip())
                        indent_str = ' ' * indent
                        
                        # Add session context
                        new_lines.append(f"{indent_str}with self.db.get_session() as session:")
                        new_lines.append(f"{indent_str}    session.query(")
                        
                        # Find the closing parenthesis and move the rest
                        j = i + 1
                        while j < len(lines) and (lines[j].strip() == '' or lines[j].startswith(' ' * (indent + 4))):
                            new_lines.append(lines[j])
                            j += 1
                        
                        # Skip the original query line
                        continue
                    else:
                        new_lines.append(line)
                
                # Write back
                with open(file_path, 'w') as f:
                    f.write('\n'.join(new_lines))
                
                print(f"✅ Fixed {file_path}")
            else:
                print(f"⏭️  No issues found in {file_path}")
        else:
            print(f"❌ File not found: {file_path}")

if __name__ == "__main__":
    fix_database_queries()
    print("🎉 Database query fixes complete!")
