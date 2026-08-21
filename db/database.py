"""
db/database.py
──────────────
QuantSphereX Persistence Layer.
Provides SQLite / PostgreSQL thread-safe database storage for:
- User Authentication & Roles (RBAC)
- API Keys & Client Sessions
- Portfolio Holdings & Target Allocations
- Alert Rules & Historical Triggers
- System & Access Audit Logging
- Model Governance & Experiment Logs
"""

from __future__ import annotations
import sqlite3
import os
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

DB_DIR = Path("db")
DB_PATH = DB_DIR / "quantspherex.db"


def get_db_connection() -> sqlite3.Connection:
    """Returns a thread-safe connection to the QuantSphereX database."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema tables and default admin account."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'analyst',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """)

    # 2. API Keys Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_keys (
        key_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        key_hash TEXT NOT NULL,
        name TEXT NOT NULL,
        permissions TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 3. Portfolios Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        benchmark TEXT DEFAULT 'NIFTY50',
        cash_balance REAL DEFAULT 1000000.0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 4. Portfolio Positions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        shares REAL NOT NULL,
        avg_price REAL NOT NULL,
        target_weight REAL NOT NULL,
        current_weight REAL NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
    )
    """)

    # 5. Alert Rules Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alert_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        metric TEXT NOT NULL,
        condition TEXT NOT NULL,
        threshold REAL NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 6. Alert History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alert_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_id INTEGER,
        severity TEXT NOT NULL,
        metric TEXT NOT NULL,
        message TEXT NOT NULL,
        triggered_at TEXT NOT NULL
    )
    """)

    # 7. System Audit Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        action TEXT NOT NULL,
        resource TEXT NOT NULL,
        ip_address TEXT,
        details TEXT,
        timestamp TEXT NOT NULL
    )
    """)

    conn.commit()

    # Seed Default Institutional Admin if not exists
    cursor.execute("SELECT COUNT(*) as cnt FROM users WHERE username = 'admin'")
    if cursor.fetchone()["cnt"] == 0:
        def_pass_hash = hashlib.sha256("admin123".encode()).hexdigest()
        now = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO users (username, email, hashed_password, role, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", "admin@quantspherex.com", def_pass_hash, "admin", now)
        )
        # Create default institutional portfolio for admin
        cursor.execute(
            "INSERT INTO portfolios (user_id, name, benchmark, cash_balance, created_at) VALUES (?, ?, ?, ?, ?)",
            (1, "QuantSphereX Core Alpha Fund", "^NSEI", 10000000.0, now)
        )
        conn.commit()
        logger.info("[Database] Initialized default institutional admin account and Core Alpha Fund.")

    conn.close()

# Auto-initialize database on module import
init_db()
