#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Diagnostic Tool for NobitexScalperPro
این فایل مشکلات کندی را شناسایی می‌کند
"""

import os
import sys
import time
import psutil
import tracemalloc
import threading
from datetime import datetime
from pathlib import Path

# شروع tracking memory
tracemalloc.start()

print("=" * 70)
print("🔍 NOBITEX SCALPER PRO - PERFORMANCE DIAGNOSTIC")
print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 1️⃣  بررسی سیستم
print("\n[1] SYSTEM INFO")
print("-" * 70)
print(f"CPU Count: {psutil.cpu_count()}")
print(f"CPU Percent: {psutil.cpu_percent(interval=1)}%")
print(f"RAM Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
print(f"RAM Used: {psutil.virtual_memory().used / (1024**3):.1f} GB")
print(f"Python Version: {sys.version}")

# 2️⃣  بررسی Process
print("\n[2] CURRENT PROCESS")
print("-" * 70)
proc = psutil.Process(os.getpid())
print(f"PID: {proc.pid}")
print(f"RSS Memory: {proc.memory_info().rss / (1024**2):.1f} MB")
print(f"VMS Memory: {proc.memory_info().vms / (1024**2):.1f} MB")
print(f"Threads: {threading.active_count()}")
print(f"Open Files: {len(proc.open_files())}")

# 3️⃣  بررسی Disk
print("\n[3] DISK SPACE")
print("-" * 70)
home = Path.home()
app_data = home / "Nobitex_AI_Trader_Pro_Data"
print(f"App Data Dir: {app_data}")
print(f"Exists: {app_data.exists()}")

if app_data.exists():
    total_size = 0
    file_count = 0
    dir_count = 0
    
    for item in app_data.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1
        elif item.is_dir():
            dir_count += 1
    
    print(f"Total Files: {file_count}")
    print(f"Total Dirs: {dir_count}")
    print(f"Total Size: {total_size / (1024**2):.1f} MB")
    
    # بزرگ‌ترین فایل‌ها
    print(f"\nLargest files:")
    largest = sorted(
        [(f.stat().st_size, str(f)) for f in app_data.rglob("*") if f.is_file()],
        reverse=True
    )[:5]
    for size, path in largest:
        print(f"  {size / (1024**2):.1f} MB - {path.split('/')[-1]}")

# 4️⃣  بررسی Network
print("\n[4] NETWORK")
print("-" * 70)
try:
    import socket
    socket.setdefaulttimeout(2)
    
    # تست اتصال به Nobitex
    print("Testing connection to Nobitex API...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start = time.time()
    result = s.connect_ex(("apiv2.nobitex.ir", 443))
    elapsed = (time.time() - start) * 1000
    s.close()
    
    if result == 0:
        print(f"✅ Nobitex API: REACHABLE ({elapsed:.0f}ms)")
    else:
        print(f"❌ Nobitex API: UNREACHABLE (timeout or blocked)")
except Exception as e:
    print(f"❌ Network test failed: {e}")

# 5️⃣  بررسی Import Performance
print("\n[5] MODULE IMPORT TIMING")
print("-" * 70)
print("Measuring import time for main application...")

try:
    start = time.time()
    # فقط imports بدون اجرای کد
    import pandas as pd
    t1 = (time.time() - start) * 1000
    print(f"pandas: {t1:.0f}ms")
    
    start = time.time()
    import numpy as np
    t2 = (time.time() - start) * 1000
    print(f"numpy: {t2:.0f}ms")
    
    start = time.time()
    import requests
    t3 = (time.time() - start) * 1000
    print(f"requests: {t3:.0f}ms")
    
    start = time.time()
    import tkinter as tk
    t4 = (time.time() - start) * 1000
    print(f"tkinter: {t4:.0f}ms")
    
    print(f"\nTotal: {t1+t2+t3+t4:.0f}ms")
except Exception as e:
    print(f"Import error: {e}")

# 6️⃣  بررسی Database
print("\n[6] DATABASE CHECK")
print("-" * 70)
try:
    import sqlite3
    
    db_files = {
        "adaptive": app_data / "adaptive_research.sqlite3",
        "unified": app_data / "unified_learning" / "NobitexUnifiedLearning.sqlite3",
        "evidence": app_data / "evidence_signal_engine" / "signal_ledger_v6.sqlite3",
    }
    
    for name, path in db_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print(f"✅ {name:12} | Size: {size_mb:8.1f} MB | {path}")
            
            # بررسی سلامت DB
            try:
                conn = sqlite3.connect(str(path), timeout=2)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
                print(f"   Tables: {table_count}")
                conn.close()
            except Exception as e:
                print(f"   ⚠️  DB Error: {e}")
        else:
            print(f"❌ {name:12} | Not found")
            
except Exception as e:
    print(f"Database check failed: {e}")

# 7️⃣  Memory Snapshot
print("\n[7] MEMORY USAGE")
print("-" * 70)
current, peak = tracemalloc.get_traced_memory()
print(f"Current Memory: {current / (1024**2):.1f} MB")
print(f"Peak Memory: {peak / (1024**2):.1f} MB")

# 8️⃣  توصیات
print("\n[8] RECOMMENDATIONS")
print("-" * 70)
recommendations = []

if psutil.virtual_memory().percent > 80:
    recommendations.append("⚠️  RAM usage is high (>80%). Close other programs.")

if file_count > 50000:
    recommendations.append("⚠️  Too many files in app data dir. Consider cleaning old data.")

if total_size / (1024**2) > 5000:
    recommendations.append("⚠️  App data dir is large (>5GB). Archive or delete old research.")

cpu = psutil.cpu_percent(interval=1)
if cpu > 70:
    recommendations.append("⚠️  CPU usage is high. App may be doing heavy computation.")

if not recommendations:
    recommendations.append("✅ No obvious performance issues detected.")
    recommendations.append("✅ Check these if program still hangs:")
    recommendations.append("   1. Are there many open trades? (slow UI updates)")
    recommendations.append("   2. Is there a slow database query?")
    recommendations.append("   3. Are there infinite loops in research workers?")
    recommendations.append("   4. Is API timeout too high? (try 3-5 seconds)")

for rec in recommendations:
    print(rec)

print("\n" + "=" * 70)
print("Diagnostic complete!")
print("=" * 70)

# Save report
report_path = Path(__file__).parent / "diagnostic_report.txt"
try:
    import io
    from contextlib import redirect_stdout
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Diagnostic Report - {datetime.now()}\n")
        f.write("=" * 70 + "\n")
        f.write(f"CPU: {psutil.cpu_percent()}%\n")
        f.write(f"RAM: {psutil.virtual_memory().percent}% used\n")
        f.write(f"App Data Size: {total_size / (1024**2):.1f} MB\n")
        f.write("\nRecommendations:\n")
        for rec in recommendations:
            f.write(f"{rec}\n")
    
    print(f"\n📝 Report saved to: {report_path}")
except Exception as e:
    print(f"Could not save report: {e}")
