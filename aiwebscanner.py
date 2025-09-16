#!/usr/bin/env python3
"""
aiwebscanner.py - Interactive web scanner with robust Ctrl+C handling.

Behavior:
- Ctrl+C during a running scan -> cancels current scan, cleans up Playwright if used,
  and returns to the REPL (you can start another scan).
- Ctrl+C once at prompt -> shows interrupt message.
- Ctrl+C twice (while stopping) -> forces exit immediately.
"""

import requests
import re
import json
import shlex
import signal
import sys
import threading
import time
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

# Optional Playwright imports (lazy)
try:
    import asyncio
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# Global coordination
GLOBAL = {
    "playwright": None,
    "browser": None,
    "running_task": None,
    "stopping": False,
    "stop_requested_at": None,
}

# Config
COMMON_DIRS = ["admin", "login", "dashboard", "config", "backup", "api", "v1", "uploads", "wp-admin", "wp-login"]
SENSITIVE_KEYWORDS = ["password", "token", "api", "key", "secret", "config", "credential"]

# Storage for results
scan_history = {}  # start_url -> data


# -------------------------
# Utility functions
# -------------------------
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def find_sensitive_keywords(text):
    if not text:
        return []
    txt = text.lower()
    found = [k for k in SENSITIVE_KEYWORDS if k in txt]
    return list(dict.fromkeys(found))


def fetch_page_static(url, session=None, timeout=10):
    try:
        s = session or requests
        resp = s.get(url, timeout=timeout, allow_redirects=True)
        return resp
    except KeyboardInterrupt:
        raise
    except Exception:
        return None


def extract_from_html(base_url, html):
    soup = BeautifulSoup(html or "", "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    js_files = [urljoin(base_url, s.get("src")) for s in soup.find_all("script", src=True)]
    css_files = [urljoin(base_url, l.get("href")) for l in soup.find_all("link", href=True)]
    forms = []
    for form in soup.find_all("form"):
        method = (form.get("method") or "GET").upper()
        action = form.get("action") or base_url
        inputs = [i.get("name") or i.get("id") or "" for i in form.find_all("input")]
        csrf = any("csrf" in (n or "").lower() for n in inputs)
        forms.append({"method": method, "action": urljoin(base_url, action), "inputs": inputs, "csrf": csrf})
    comments = re.findall(r'<!--(.*?)-->', html or "", re.S)
    links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]
    return {
        "title": title,
        "js_files": list(dict.fromkeys(js_files)),
        "css_files": list(dict.fromkeys(css_files)),
        "forms": forms,
        "comments": comments,
        "links": links
    }


def parse_js_for_endpoints(js_text):
    endpoints = set()
    # capture explicit fetch/axios patterns and path-like strings
    try:
        for m in re.findall(r"""(?:fetch|axios(?:\.(?:get|post|put|delete))?)\s*\(\s*['"]([^'"]+)['"]""", js_text, re.I):
            if m.startswith("/"):
                endpoints.add(m)
        for m in re.findall(r"""['"](/(?:[a-zA-Z0-9_\-./]+))['"]""", js_text):
            endpoints.add(m)
        if "graphql" in (js_text or "").lower():
            endpoints.add("/graphql")
    except Exception:
        pass
    return list(endpoints)


# -------------------------
# Playwright helpers (dynamic)
# -------------------------
async def _start_playwright_browser():
    if not PLAYWRIGHT_AVAILABLE:
        return None, None
    if GLOBAL["playwright"] is None:
        GLOBAL["playwright"] = await async_playwright().start()
    browser = await GLOBAL["playwright"].chromium.launch(headless=True)
    GLOBAL["browser"] = browser
    return GLOBAL["playwright"], browser


async def _close_playwright_browser():
    try:
        if GLOBAL.get("browser"):
            await GLOBAL["browser"].close()
            GLOBAL["browser"] = None
        if GLOBAL.get("playwright"):
            await GLOBAL["playwright"].stop()
            GLOBAL["playwright"] = None
    except Exception:
        pass


async def dynamic_fetch(url, wait_ms=2500):
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not available")
    p, browser = await _start_playwright_browser()
    page = await browser.new_page()
    api_calls = []

    def _on_request(request):
        try:
            if request.resource_type in ("xhr", "fetch"):
                api_calls.append(request.url)
        except Exception:
            pass

    page.on("request", _on_request)

    try:
        await page.goto(url, timeout=30000)
        await page.wait_for_timeout(wait_ms)
        content = await page.content()
    except Exception:
        try:
            content = await page.content()
        except Exception:
            content = ""
    finally:
        try:
            await page.close()
        except Exception:
            pass
    return content, api_calls


# -------------------------
# Crawl static (recursive)
# -------------------------
def crawl_static(start_url, depth=2, session=None):
    to_visit = [(start_url, 0)]
    visited = set()
    session = session or requests
    while to_visit:
        if GLOBAL["stopping"]:
            safe_print("[!] Crawl: stop requested, breaking out.")
            break
        url, d = to_visit.pop(0)
        if url in visited or d > depth:
            continue
        visited.add(url)
        safe_print(f"[>] Fetching {url} (depth {d})")
        try:
            resp = fetch_page_static(url, session=session)
            if resp is None:
                safe_print(f"[!] Failed to fetch {url}")
                continue
            base = resp.url
            page_data = extract_from_html(base, resp.text)
            sd = scan_history.setdefault(start_url, {"pages": {}, "start_url": start_url})
            sd["pages"][base] = {
                "status": resp.status_code,
                "headers": dict(resp.headers),
                "title": page_data["title"],
                "forms": page_data["forms"],
                "js_files": page_data["js_files"],
                "css_files": page_data["css_files"],
                "comments": page_data["comments"],
                "keywords": find_sensitive_keywords(resp.text)
            }
            for l in page_data["links"]:
                parsed_base = urlparse(start_url)
                parsed_l = urlparse(l)
                if parsed_l.netloc == parsed_base.netloc:
                    to_visit.append((l, d + 1))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            safe_print(f"[!] Error fetching {url}: {e}")
            continue


def scan_common_dirs(base_url):
    found = []
    for d in COMMON_DIRS:
        if GLOBAL["stopping"]:
            break
        target = urljoin(base_url.rstrip("/") + "/", d)
        try:
            r = requests.head(target, allow_redirects=True, timeout=6)
            if r is not None and r.status_code < 400:
                found.append((target, r.status_code))
        except KeyboardInterrupt:
            raise
        except Exception:
            continue
    return found


# -------------------------
# High level scan
# -------------------------
def scan_target(target_url, depth=2, dynamic=False):
    if GLOBAL["stopping"]:
        safe_print("[!] Stop already requested - aborting new scan.")
        return
    safe_print(f"[+] Starting scan: {target_url} (depth={depth}, dynamic={dynamic})")
    if not target_url.startswith("http://") and not target_url.startswith("https://"):
        target_url = "http://" + target_url
    try:
        crawl_static(target_url, depth=depth)
    except KeyboardInterrupt:
        safe_print("[!] Scan interrupted (static).")
    except Exception as e:
        safe_print(f"[!] Static scan error: {e}")

    # parse JS for endpoints
    sd = scan_history.get(target_url, {"pages": {}, "start_url": target_url})
    for page_url, pdata in list(sd["pages"].items()):
        if GLOBAL["stopping"]:
            safe_print("[!] Stop requested - halting JS parsing.")
            break
        js_files = pdata.get("js_files", [])
        endpoints = set(pdata.get("api_endpoints", []))
        for js in js_files:
            try:
                jr = requests.get(js, timeout=8)
                if jr and jr.status_code == 200:
                    for e in parse_js_for_endpoints(jr.text):
                        if e.startswith("/"):
                            endpoints.add(urljoin(page_url, e))
                        else:
                            endpoints.add(e)
            except KeyboardInterrupt:
                raise
            except Exception:
                continue
        pdata["api_endpoints"] = sorted(endpoints)

    # dynamic
    if dynamic:
        if not PLAYWRIGHT_AVAILABLE:
            safe_print("[!] Playwright not available - dynamic scan skipped.")
        else:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                GLOBAL["running_task"] = loop.create_task(dynamic_fetch(target_url))
                content, api_calls = loop.run_until_complete(GLOBAL["running_task"])
                # merge dynamic results into start page entry
                pdata = scan_history.setdefault(target_url, {"pages": {}, "start_url": target_url})["pages"].setdefault(target_url, {})
                ddata = extract_from_html(target_url, content)
                pdata.setdefault("js_files", [])
                pdata["js_files"] = sorted(list(dict.fromkeys(pdata.get("js_files", []) + ddata.get("js_files", []))))
                pdata.setdefault("api_endpoints", [])
                pdata["api_endpoints"] = sorted(list(dict.fromkeys(pdata.get("api_endpoints", []) + api_calls)))
                pdata.setdefault("comments", [])
                pdata["comments"] = pdata.get("comments", []) + ddata.get("comments", [])
                pdata["keywords"] = find_sensitive_keywords(content)
            except KeyboardInterrupt:
                safe_print("[!] Dynamic scan interrupted.")
            except Exception as e:
                safe_print(f"[!] Dynamic scan error: {e}")
            finally:
                try:
                    loop.run_until_complete(_close_playwright_browser())
                except Exception:
                    pass
                try:
                    loop.close()
                except Exception:
                    pass
                GLOBAL["running_task"] = None

    # common dirs
    try:
        cd = scan_common_dirs(target_url)
        if cd:
            scan_history.setdefault(target_url, {}).setdefault("common_dirs", []).extend(cd)
    except KeyboardInterrupt:
        safe_print("[!] Common-dir scan interrupted.")
    except Exception:
        pass

    safe_print(f"[+] Scan finished for: {target_url}")
    # reset stopping flag if scan finished naturally
    GLOBAL["stopping"] = False
    GLOBAL["stop_requested_at"] = None


# -------------------------
# AI heuristic summary
# -------------------------
def ai_help_print(target_url):
    data = scan_history.get(target_url)
    if not data:
        safe_print("[!] No scan data found for that URL.")
        return
    safe_print("\n--- AI Vulnerability Summary (heuristic) ---")
    for page, pdata in data["pages"].items():
        headers = pdata.get("headers", {})
        missing = [h for h in ["Content-Security-Policy", "X-Frame-Options", "Strict-Transport-Security", "X-Content-Type-Options", "Referrer-Policy"] if h not in headers]
        if missing:
            safe_print(f"[!] {page} missing headers: {', '.join(missing)}")
        for f in pdata.get("forms", []):
            if not f.get("csrf"):
                safe_print(f"[!] Form without CSRF: {f.get('method')} action={f.get('action')}")
        if pdata.get("keywords"):
            safe_print(f"[!] Sensitive keywords: {', '.join(pdata.get('keywords', []))}")
        if pdata.get("api_endpoints"):
            safe_print(f"[!] API endpoints ({len(pdata.get('api_endpoints'))}): {', '.join(pdata.get('api_endpoints')[:20])}")
    if data.get("common_dirs"):
        for cd, sts in data["common_dirs"]:
            safe_print(f"[!] Common dir found: {cd} (status {sts})")
    safe_print("[+] Heuristic summary complete.\n")


# -------------------------
# Signal handling
# -------------------------
def handle_sigint(signum, frame):
    now = time.time()
    # If stop already requested recently -> force exit
    if GLOBAL["stopping"]:
        safe_print("\n[!] Second Ctrl+C detected — forcing exit.")
        try:
            # attempt final cleanup
            if PLAYWRIGHT_AVAILABLE:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_close_playwright_browser())
                loop.close()
        except Exception:
            pass
        sys.exit(1)

    # First Ctrl+C: request stop and cancel running async task if any
    GLOBAL["stopping"] = True
    GLOBAL["stop_requested_at"] = now
    safe_print("\n[!] SIGINT received — stopping current scan gracefully...")
    # cancel running async task
    try:
        task = GLOBAL.get("running_task")
        if task and not task.done():
            try:
                task.cancel()
            except Exception:
                pass
    except Exception:
        pass

    # start a background thread that will reset stopping flag if nothing else happens
    def reset_stop_flag_later():
        # give a grace period for cleanup
        time.sleep(3)
        # if stopping still true and no task running -> allow further scans
        if GLOBAL["stopping"]:
            GLOBAL["stopping"] = False
            GLOBAL["stop_requested_at"] = None
            safe_print("[!] Stop cleared — you may run new scans.")
    t = threading.Thread(target=reset_stop_flag_later, daemon=True)
    t.start()


signal.signal(signal.SIGINT, handle_sigint)


# -------------------------
# REPL
# -------------------------
def repl():
    safe_print("=== AI Web Scanner (Interactive Mode) ===")
    safe_print("Commands:")
    safe_print("  scan <url> [--depth N] [--dynamic]  : Scan target")
    safe_print("  show forms                          : Show forms from last scan")
    safe_print("  show js                             : Show JS files from last scan")
    safe_print("  show api                            : Show API endpoints from last scan")
    safe_print("  ai help                             : Heuristic AI summary")
    safe_print("  exit                                : Quit\n")

    last_target = None
    while True:
        try:
            line = input("aiwebscanner> ").strip()
        except EOFError:
            safe_print("\nExiting.")
            break
        except KeyboardInterrupt:
            safe_print("\n[!] Interrupted at prompt. Type 'exit' to quit.")
            continue

        if not line:
            continue
        parts = shlex.split(line)
        cmd = parts[0].lower()

        if cmd in ("exit", "quit"):
            safe_print("Exiting.")
            # ensure playwright cleanup before exit
            if PLAYWRIGHT_AVAILABLE:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(_close_playwright_browser())
                    loop.close()
                except Exception:
                    pass
            break

        if cmd == "scan":
            if len(parts) < 2:
                safe_print("[!] Usage: scan <url> [--depth N] [--dynamic]")
                continue
            target = parts[1]
            depth = 2
            dynamic = False
            i = 2
            while i < len(parts):
                p = parts[i]
                if p == "--depth" and i + 1 < len(parts):
                    try:
                        depth = int(parts[i + 1]); i += 2; continue
                    except Exception:
                        pass
                if p == "--dynamic":
                    dynamic = True
                i += 1
            last_target = target
            # reset stopping flag before starting a new scan
            GLOBAL["stopping"] = False
            GLOBAL["stop_requested_at"] = None
            try:
                scan_target(target, depth=depth, dynamic=dynamic)
            except KeyboardInterrupt:
                safe_print("\n[!] Scan interrupted by user (outer).")
                GLOBAL["stopping"] = False
            continue

        if cmd == "show":
            if len(parts) < 2:
                safe_print("[!] Usage: show <forms|js|api>")
                continue
            what = parts[1].lower()
            if not last_target or last_target not in scan_history:
                safe_print("[!] No scan data available. Run 'scan <url>' first.")
                continue
            data = scan_history[last_target]
            pages = data.get("pages", {})
            if what == "forms":
                for page, pdata in pages.items():
                    fs = pdata.get("forms", [])
                    if fs:
                        safe_print(f"\n[== forms for {page} ==]")
                        for f in fs:
                            safe_print(json.dumps(f, indent=2))
            elif what == "js":
                for page, pdata in pages.items():
                    js = pdata.get("js_files", [])
                    if js:
                        safe_print(f"\n[== js files for {page} ==]")
                        for j in js:
                            safe_print(j)
            elif what == "api":
                for page, pdata in pages.items():
                    api = pdata.get("api_endpoints", [])
                    if api:
                        safe_print(f"\n[== api endpoints for {page} ==]")
                        for a in api:
                            safe_print(a)
            else:
                safe_print("[!] Unknown show target. Use 'show forms|js|api'.")
            continue

        if cmd == "ai":
            if len(parts) >= 2 and parts[1] == "help":
                if not last_target:
                    safe_print("[!] No previous target. Use 'scan <url>' first.")
                else:
                    ai_help_print(last_target)
            else:
                safe_print("[!] Usage: ai help")
            continue

        safe_print("[!] Unknown command. Use 'scan', 'show', 'ai help', or 'exit'.")


if __name__ == "__main__":
    repl()
