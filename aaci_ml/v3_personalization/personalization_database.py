"""
AACI V3.1 – SQLite Profile Persistence Layer
============================================
Atomberg Adaptive Comfort Intelligence (AACI)

Provides persistent SQLite storage for individual occupant comfort profiles
and interaction audit logs.

Schema:
  - occupant_profiles:
      user_id (TEXT PRIMARY KEY)
      name (TEXT)
      feedback_count (INTEGER)
      logit_cold (REAL)
      logit_neutral (REAL)
      logit_hot (REAL)
      setpoint_shift_c (REAL)
      speed_preference_offset (INTEGER)
      white_noise_affinity (INTEGER)
      created_at (REAL)
      updated_at (REAL)

  - feedback_history:
      id (INTEGER PRIMARY KEY AUTOINCREMENT)
      user_id (TEXT)
      timestamp (REAL)
      feedback_type (TEXT)
      reason (TEXT)
      air_temp_c (REAL)
      relative_humidity (REAL)
      current_speed (INTEGER)
      manual_new_speed (INTEGER)
      update_detail (TEXT)
"""

import os
import sqlite3
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any

_DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aaci_profiles.db")

class PersonalizationDatabase:
    """Manages SQLite persistence for occupant profiles and interaction history."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS occupant_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    feedback_count INTEGER DEFAULT 0,
                    logit_cold REAL DEFAULT 0.0,
                    logit_neutral REAL DEFAULT 0.0,
                    logit_hot REAL DEFAULT 0.0,
                    setpoint_shift_c REAL DEFAULT 0.0,
                    speed_preference_offset INTEGER DEFAULT 0,
                    white_noise_affinity INTEGER DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp REAL,
                    feedback_type TEXT,
                    reason TEXT,
                    air_temp_c REAL,
                    relative_humidity REAL,
                    current_speed INTEGER,
                    manual_new_speed INTEGER,
                    update_detail TEXT,
                    FOREIGN KEY (user_id) REFERENCES occupant_profiles(user_id)
                )
            """)
            conn.commit()

    def save_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Inserts or updates an occupant profile."""
        now = time.time()
        user_id = str(profile_data.get("user_id", "")).strip()
        if not user_id:
            return False

        logit_offsets = profile_data.get("logit_offsets", [0.0, 0.0, 0.0])
        try:
            lc = float(logit_offsets[0])
            ln = float(logit_offsets[1])
            lh = float(logit_offsets[2])
        except (IndexError, TypeError, ValueError):
            lc, ln, lh = 0.0, 0.0, 0.0

        name = str(profile_data.get("name", user_id))
        feedback_count = int(profile_data.get("feedback_count", 0))
        setpoint_shift_c = float(profile_data.get("setpoint_shift_c", 0.0))
        speed_offset = int(profile_data.get("speed_preference_offset", 0))
        white_noise = 1 if profile_data.get("white_noise_affinity", False) else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO occupant_profiles (
                    user_id, name, feedback_count,
                    logit_cold, logit_neutral, logit_hot,
                    setpoint_shift_c, speed_preference_offset,
                    white_noise_affinity, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    feedback_count = excluded.feedback_count,
                    logit_cold = excluded.logit_cold,
                    logit_neutral = excluded.logit_neutral,
                    logit_hot = excluded.logit_hot,
                    setpoint_shift_c = excluded.setpoint_shift_c,
                    speed_preference_offset = excluded.speed_preference_offset,
                    white_noise_affinity = excluded.white_noise_affinity,
                    updated_at = excluded.updated_at
            """, (
                user_id, name, feedback_count,
                lc, ln, lh, setpoint_shift_c,
                speed_offset, white_noise, now, now
            ))
            conn.commit()
        return True

    def load_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Loads an occupant profile by user_id. Returns None if not found or corrupted."""
        user_id = str(user_id).strip()
        if not user_id:
            return None

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM occupant_profiles WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                return {
                    "user_id": row["user_id"],
                    "name": row["name"],
                    "feedback_count": int(row["feedback_count"]),
                    "logit_offsets": [float(row["logit_cold"]), float(row["logit_neutral"]), float(row["logit_hot"])],
                    "setpoint_shift_c": float(row["setpoint_shift_c"]),
                    "speed_preference_offset": int(row["speed_preference_offset"]),
                    "white_noise_affinity": bool(row["white_noise_affinity"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        except Exception:
            return None

    def load_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Loads all profiles into a dictionary keyed by user_id."""
        profiles = {}
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM occupant_profiles")
                for row in cursor.fetchall():
                    try:
                        uid = row["user_id"]
                        profiles[uid] = {
                            "user_id": uid,
                            "name": row["name"],
                            "feedback_count": int(row["feedback_count"]),
                            "logit_offsets": [float(row["logit_cold"]), float(row["logit_neutral"]), float(row["logit_hot"])],
                            "setpoint_shift_c": float(row["setpoint_shift_c"]),
                            "speed_preference_offset": int(row["speed_preference_offset"]),
                            "white_noise_affinity": bool(row["white_noise_affinity"]),
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        }
                    except Exception:
                        continue
        except Exception:
            pass
        return profiles

    def record_history(
        self,
        user_id: str,
        feedback_type: str,
        reason: Optional[str],
        air_temp_c: float,
        relative_humidity: float,
        current_speed: int,
        manual_new_speed: Optional[int],
        update_detail: str,
        timestamp: Optional[float] = None
    ) -> bool:
        """Appends a feedback observation to the audit log."""
        ts = timestamp if timestamp is not None else time.time()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO feedback_history (
                        user_id, timestamp, feedback_type, reason,
                        air_temp_c, relative_humidity, current_speed,
                        manual_new_speed, update_detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, ts, feedback_type, reason,
                    air_temp_c, relative_humidity, current_speed,
                    manual_new_speed, update_detail
                ))
                conn.commit()
            return True
        except Exception:
            return False

    def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves interaction history for a specific occupant."""
        results = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM feedback_history 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (user_id, limit))
                for row in cursor.fetchall():
                    results.append(dict(row))
        except Exception:
            pass
        return results

    def delete_profile(self, user_id: str) -> bool:
        """Deletes an occupant profile and history (factory reset)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feedback_history WHERE user_id = ?", (user_id,))
                cursor.execute("DELETE FROM occupant_profiles WHERE user_id = ?", (user_id,))
                conn.commit()
            return True
        except Exception:
            return False

