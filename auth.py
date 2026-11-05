import sqlite3
import hashlib
import hmac
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "users.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        600_000
    ).hex()

    return password_hash, salt


def register_user(username, email, password):
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Please fill in every field."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    password_hash, salt = hash_password(password)

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, salt)
                VALUES (?, ?, ?, ?)
                """,
                (username, email, password_hash, salt)
            )

        return True, "Registration successful. Please log in."

    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def login_user(username_or_email, password):
    identifier = username_or_email.strip()

    with get_connection() as conn:
        user = conn.execute(
            """
            SELECT id, username, email, password_hash, salt
            FROM users
            WHERE username = ? COLLATE NOCASE
               OR email = ? COLLATE NOCASE
            """,
            (identifier, identifier)
        ).fetchone()

    if user is None:
        return False, None

    user_id, username, email, stored_hash, salt = user
    calculated_hash, _ = hash_password(password, salt)

    if hmac.compare_digest(calculated_hash, stored_hash):
        return True, {
            "id": user_id,
            "username": username,
            "email": email
        }

    return False, None


initialize_database()