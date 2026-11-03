"""
AACI V3.1 – Individual Thermal Comfort Personalization Engine
=============================================================
Atomberg Adaptive Comfort Intelligence (AACI)

Architecture:
  Environmental Inputs
    → Generic Thermal Comfort Model (V2 XGBoost)
    → Individual Personalization Layer (Bayesian-Inspired Adaptive Logit Offsets)
    → Confidence & Motor Feedback Evaluation (AACIFanControllerV3)
    → Deterministic BLDC Motor Command

Reliability Updates in V3.1:
  - SQLite persistence via PersonalizationDatabase: profiles survive application restarts.
  - Separate acoustic affinity tracking ('WHITE_NOISE') from thermal discomfort.
  - Controlled speed preference adjustments for reasonless manual overrides (no blind 'TOO_HOT' assumption).
  - Calibrated setpoint coupling modes: 'COMBINED', 'LOGIT_ONLY', 'SETPOINT_ONLY'.
  - Isolated memory and database storage per occupant ID.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

from personalization_database import PersonalizationDatabase

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

class UserComfortProfile:
    """
    Stores and adapts individual thermal comfort preferences.
    """
    def __init__(self, user_id: str, name: Optional[str] = None):
        self.user_id = str(user_id).strip()
        self.name = name or self.user_id
        self.feedback_count = 0
        
        # Logit offset vector for [Cold, Neutral, Hot]
        self.logit_offsets = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Thermal setpoint bias (Celsius shift from population mean)
        # > 0 means occupant prefers warmer rooms (chill-sensitive)
        # < 0 means occupant prefers cooler rooms (runs hot / warm-sensitive)
        self.setpoint_shift_c = 0.0
        
        # Fan speed baseline preference offset (-2 to +2 steps)
        self.speed_preference_offset = 0
        
        # Acoustic context flag
        self.white_noise_affinity = False
        
        # Timestamped interaction history (in-memory cache)
        self.interaction_log: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "feedback_count": int(self.feedback_count),
            "logit_offsets": [round(float(x), 4) for x in self.logit_offsets],
            "setpoint_shift_c": round(float(self.setpoint_shift_c), 2),
            "speed_preference_offset": int(self.speed_preference_offset),
            "white_noise_affinity": bool(self.white_noise_affinity),
            "recent_interactions": len(self.interaction_log),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserComfortProfile":
        """Reconstructs profile from persistent dictionary."""
        prof = cls(user_id=data["user_id"], name=data.get("name"))
        prof.feedback_count = int(data.get("feedback_count", 0))
        lo = data.get("logit_offsets", [0.0, 0.0, 0.0])
        prof.logit_offsets = np.array([float(lo[0]), float(lo[1]), float(lo[2])], dtype=float)
        prof.setpoint_shift_c = float(data.get("setpoint_shift_c", 0.0))
        prof.speed_preference_offset = int(data.get("speed_preference_offset", 0))
        prof.white_noise_affinity = bool(data.get("white_noise_affinity", False))
        return prof


class PersonalizationEngine:
    """
    Personalization layer that adapts generic ML probabilities to an individual.
    Uses Bayesian-inspired adaptive logit offsets and SQLite profile persistence.
    """
    
    CLASS_NAMES = ["Cold", "Neutral", "Hot"]
    
    def __init__(self, db_path: Optional[str] = None, adaptation_mode: str = "COMBINED"):
        """
        Parameters:
        -----------
        db_path : Optional[str]
            Path to SQLite database file. If None, uses default 'aaci_profiles.db'.
        adaptation_mode : str
            - 'COMBINED': Applies both logit offsets and calibrated setpoint bias.
            - 'LOGIT_ONLY': Applies only logit offset adjustments.
            - 'SETPOINT_ONLY': Applies only setpoint shift temperature scaling.
        """
        self.adaptation_mode = adaptation_mode
        self.db = PersonalizationDatabase(db_path=db_path)
        self.profiles: Dict[str, UserComfortProfile] = {}
        
        # Load any existing persisted profiles into memory cache on initialization
        persisted = self.db.load_all_profiles()
        for uid, pdata in persisted.items():
            self.profiles[uid] = UserComfortProfile.from_dict(pdata)

    def get_or_create_profile(self, user_id: str, name: Optional[str] = None) -> UserComfortProfile:
        uid = str(user_id).strip()
        if uid not in self.profiles:
            # Check if exists in DB
            db_data = self.db.load_profile(uid)
            if db_data:
                self.profiles[uid] = UserComfortProfile.from_dict(db_data)
            else:
                new_prof = UserComfortProfile(user_id=uid, name=name)
                self.profiles[uid] = new_prof
                self.db.save_profile(new_prof.to_dict())
        return self.profiles[uid]

    def personalize_probabilities(
        self,
        generic_probs: Dict[str, float],
        user_id: str,
        air_temp_c: float = 24.0,
    ) -> Dict[str, Any]:
        """
        Transforms generic population probabilities into user-personalized probabilities
        using the individual's learned logit offset vector and temperature context.
        """
        profile = self.get_or_create_profile(user_id)
        
        # Convert generic probs to array [p_cold, p_neutral, p_hot]
        p_raw = np.array([
            max(1e-4, float(generic_probs.get("Cold", 0.3333))),
            max(1e-4, float(generic_probs.get("Neutral", 0.3333))),
            max(1e-4, float(generic_probs.get("Hot", 0.3333))),
        ], dtype=float)
        p_raw = p_raw / p_raw.sum()
        
        # Base population logits
        logits = np.log(p_raw)
        adjusted_logits = logits.copy()
        
        # 1. Apply logit offsets (if enabled)
        if self.adaptation_mode in ("COMBINED", "LOGIT_ONLY"):
            adjusted_logits += profile.logit_offsets
            
        # 2. Apply temperature setpoint bias (if enabled)
        # Contextual physical scaling: shift towards Cold if occupant prefers warmer ambient
        if self.adaptation_mode in ("COMBINED", "SETPOINT_ONLY"):
            if profile.setpoint_shift_c != 0.0:
                temp_delta = profile.setpoint_shift_c
                # In COMBINED mode, scale factor is calibrated to 0.10 to prevent excessive double-counting
                gamma = 0.10 if self.adaptation_mode == "COMBINED" else 0.20
                adjusted_logits[0] += gamma * temp_delta
                adjusted_logits[2] -= gamma * temp_delta
            
        p_personalized = softmax(adjusted_logits)
        
        class_id = int(np.argmax(p_personalized))
        predicted_class = self.CLASS_NAMES[class_id]
        confidence = float(p_personalized[class_id])
        
        return {
            "personalized_class": predicted_class,
            "class_id": class_id,
            "probabilities": {
                "Cold": round(float(p_personalized[0]), 4),
                "Neutral": round(float(p_personalized[1]), 4),
                "Hot": round(float(p_personalized[2]), 4),
            },
            "confidence": round(confidence, 4),
            "generic_class": self.CLASS_NAMES[int(np.argmax(p_raw))],
            "generic_confidence": round(float(np.max(p_raw)), 4),
            "user_id": user_id,
            "feedback_count": profile.feedback_count,
            "profile_summary": profile.to_dict(),
        }

    def record_feedback(
        self,
        user_id: str,
        feedback_type: str,
        air_temp_c: float,
        relative_humidity: float,
        current_speed: int,
        reason: Optional[str] = None,
        manual_new_speed: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Updates the user's comfort profile based on explicit feedback or manual adjustment.
        Saves changes immediately to SQLite persistence.
        """
        profile = self.get_or_create_profile(user_id)
        now = timestamp if timestamp is not None else time.time()
        
        # Hyperbolic decaying learning rate schedule: eta(N) = 0.40 / (1 + 0.15N)
        eta = 0.40 / (1.0 + 0.15 * profile.feedback_count)
        
        update_applied = False
        update_detail = ""
        
        # 1. Explicit thermal votes
        if feedback_type == "FEELING_HOT":
            profile.logit_offsets += np.array([-0.6 * eta, -0.2 * eta, +0.8 * eta])
            profile.setpoint_shift_c -= 0.25 * eta  # User prefers cooler environment
            update_applied = True
            update_detail = "Increased Hot class affinity; shifted setpoint cooler."
            
        elif feedback_type == "FEELING_COLD":
            profile.logit_offsets += np.array([+0.8 * eta, -0.2 * eta, -0.6 * eta])
            profile.setpoint_shift_c += 0.25 * eta  # User prefers warmer environment
            update_applied = True
            update_detail = "Increased Cold class affinity; shifted setpoint warmer."
            
        elif feedback_type == "FEELING_COMFORTABLE":
            profile.logit_offsets += np.array([-0.2 * eta, +0.5 * eta, -0.2 * eta])
            update_applied = True
            update_detail = "Reinforced Neutral comfort class for current ambient regime."

        # 2. Manual speed changes with structured reason parsing
        elif feedback_type == "MANUAL_SPEED_CHANGE":
            delta_speed = (manual_new_speed - current_speed) if manual_new_speed is not None else 0
            reason_upper = (reason or "").strip().upper()
            
            if "NOISE" in reason_upper or "SLEEP_SOUND" in reason_upper:
                # Acoustic preference: NOT thermal discomfort!
                profile.white_noise_affinity = True
                update_applied = True
                update_detail = "Logged white noise / acoustic preference. Thermal comfort model NOT distorted."
                
            elif "TOO_HOT" in reason_upper:
                profile.logit_offsets += np.array([-0.4 * eta, -0.1 * eta, +0.5 * eta])
                profile.setpoint_shift_c -= 0.2 * eta
                profile.speed_preference_offset = min(2, profile.speed_preference_offset + 1)
                update_applied = True
                update_detail = "Manual speed increase due to warmth: adapted Hot affinity."
                
            elif "TOO_COLD" in reason_upper or "DRAFT" in reason_upper:
                profile.logit_offsets += np.array([+0.5 * eta, -0.1 * eta, -0.4 * eta])
                profile.setpoint_shift_c += 0.2 * eta
                profile.speed_preference_offset = max(-2, profile.speed_preference_offset - 1)
                update_applied = True
                update_detail = "Manual speed decrease due to chill: adapted Cold affinity."
                
            elif "STUFFY" in reason_upper:
                profile.speed_preference_offset = min(2, profile.speed_preference_offset + 1)
                update_applied = True
                update_detail = "Logged stuffy air sensation: adjusted baseline speed preference offset."
                
            elif not reason_upper and delta_speed != 0:
                # Manual change without explicit reason: DO NOT assume hot!
                # Only adjust speed preference offset gently without corrupting thermal logit affinities
                if delta_speed > 0:
                    profile.speed_preference_offset = min(2, profile.speed_preference_offset + 1)
                    update_detail = "Unprompted manual increase: adjusted speed preference offset without thermal bias."
                else:
                    profile.speed_preference_offset = max(-2, profile.speed_preference_offset - 1)
                    update_detail = "Unprompted manual decrease: adjusted speed preference offset without thermal bias."
                update_applied = True

        # Clamp offsets to prevent extreme distortion
        profile.logit_offsets = np.clip(profile.logit_offsets, -1.8, 1.8)
        profile.setpoint_shift_c = np.clip(profile.setpoint_shift_c, -3.0, 3.0)
        
        if update_applied:
            profile.feedback_count += 1
            log_entry = {
                "timestamp": now,
                "feedback_type": feedback_type,
                "reason": reason,
                "air_temp_c": air_temp_c,
                "relative_humidity": relative_humidity,
                "current_speed": current_speed,
                "manual_new_speed": manual_new_speed,
                "update_detail": update_detail,
                "updated_offsets": list(profile.logit_offsets.round(3)),
            }
            profile.interaction_log.append(log_entry)
            
            # Persist immediately to SQLite database
            self.db.save_profile(profile.to_dict())
            self.db.record_history(
                user_id=user_id,
                feedback_type=feedback_type,
                reason=reason,
                air_temp_c=air_temp_c,
                relative_humidity=relative_humidity,
                current_speed=current_speed,
                manual_new_speed=manual_new_speed,
                update_detail=update_detail,
                timestamp=now,
            )
            
        return {
            "user_id": user_id,
            "update_applied": update_applied,
            "feedback_count": profile.feedback_count,
            "update_detail": update_detail,
            "current_profile": profile.to_dict(),
        }

    def reset_profile(self, user_id: str) -> bool:
        """Resets an occupant profile to factory default in memory and database."""
        uid = str(user_id).strip()
        if uid in self.profiles:
            del self.profiles[uid]
        return self.db.delete_profile(uid)
