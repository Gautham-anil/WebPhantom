#!/usr/bin/env python3
import requests
import signal
import sys
import cmd
from bs4 import BeautifulSoup

scan_results = {}

class AIWebScanner(cmd.Cmd):
    intro = "Welcome to WebPhantom (AI-powered Web Scanner). Type help or ? to list commands.\n"
    prompt = "aiwebscanner> "

    def __init__(self):
        super(AIWebScanner, self).__init__()
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self, sig, frame):
        print("\n[!] SIGINT received — stopping gracefully...")
        sys.exit(0)

    def do_scan(self, url):
        """Scan a target URL for basic vulnerabilities"""
        if not url:
            print("[!] Usage: scan <url>")
            return
        print(f"[+] Scanning: {url}")
        result = self.basic_scan(url)
        scan_results[url] = result
        print("[+] Scan completed.")

    def basic_scan(self, url):
        findings = {}
        try:
            r = requests.get(url, timeout=10)
            findings["status_code"] = r.status_code

            if "php?id=" in r.text.lower():
                findings["sqli"] = "Potential SQLi found"

            if "<script>" in r.text.lower():
                findings["xss"] = "Potential XSS found"

            soup = BeautifulSoup(r.text, "html.parser")
            forms = soup.find_all("form")
            if forms:
                findings["forms"] = f"{len(forms)} form(s) detected"
        except Exception as e:
            findings["error"] = str(e)
        return findings

    def do_show(self, url):
        """Show scan results for a given URL"""
        if url in scan_results:
            for k, v in scan_results[url].items():
                print(f"{k}: {v}")
        else:
            print("[!] No scan data found for that URL.")

    def do_ai(self, arg):
        """AI assistant for deeper insights. Usage: ai help"""
        if arg == "help":
            print("[AI] Based on scans, I suggest:")
            for url, findings in scan_results.items():
                print(f"\nTarget: {url}")
                if "sqli" in findings:
                    print("  - Try sqlmap for SQL Injection exploitation.")
                if "xss" in findings:
                    print("  - Test XSS using payload lists.")
                if "forms" in findings:
                    print("  - Analyze forms for CSRF tokens.")
                if not findings:
                    print("  - No major findings, try a deeper scan.")
        else:
            print("[!] Unknown AI command. Try: ai help")

    def do_exit(self, arg):
        """Exit the tool"""
        print("[+] Exiting WebPhantom. Goodbye!")
        return True


if __name__ == "__main__":
    AIWebScanner().cmdloop()
