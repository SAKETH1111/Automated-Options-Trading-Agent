#!/usr/bin/env python3
"""
Simple webhook server for GitHub deployment
Run this on the droplet to enable webhook-based deployments
"""

import os
import json
import hmac
import hashlib
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/webhook':
            self.handle_webhook()
        else:
            self.send_error(404)
    
    def handle_webhook(self):
        try:
            # Get content length
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read the payload
            payload = self.rfile.read(content_length)
            
            # Verify GitHub signature (optional but recommended)
            github_signature = self.headers.get('X-Hub-Signature-256', '')
            if github_signature:
                expected_signature = 'sha256=' + hmac.new(
                    os.environ.get('GITHUB_WEBHOOK_SECRET', '').encode(),
                    payload,
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(github_signature, expected_signature):
                    self.send_error(401, "Unauthorized")
                    return
            
            # Parse the payload
            event_data = json.loads(payload.decode('utf-8'))
            
            # Check if this is a push to main branch
            if (event_data.get('ref') == 'refs/heads/main' and 
                event_data.get('head_commit')):
                
                print(f"🔄 Webhook triggered by push from {event_data['head_commit']['author']['name']}")
                
                # Run deployment script
                result = subprocess.run(
                    ['/opt/trading-agent/scripts/github_deploy.sh'],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        'status': 'success',
                        'message': 'Deployment completed successfully',
                        'output': result.stdout
                    }
                else:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        'status': 'error',
                        'message': 'Deployment failed',
                        'error': result.stderr
                    }
                
                self.wfile.write(json.dumps(response).encode())
            else:
                # Not a push to main, just acknowledge
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ignored'}).encode())
                
        except Exception as e:
            print(f"❌ Webhook error: {e}")
            self.send_error(500, str(e))
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Check service status
            try:
                result = subprocess.run(['systemctl', 'is-active', 'trading-agent'], 
                                      capture_output=True, text=True)
                service_status = result.stdout.strip()
                
                response = {
                    'status': 'ok',
                    'service': service_status,
                    'timestamp': subprocess.run(['date', '-Iseconds'], 
                                              capture_output=True, text=True).stdout.strip()
                }
            except Exception as e:
                response = {'status': 'error', 'error': str(e)}
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def main():
    port = int(os.environ.get('WEBHOOK_PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), WebhookHandler)
    print(f"🚀 Webhook server starting on port {port}")
    print(f"📡 Webhook URL: http://45.55.150.19:{port}/webhook")
    print(f"🔍 Health check: http://45.55.150.19:{port}/health")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Webhook server stopped")
        server.shutdown()

if __name__ == '__main__':
    main()
