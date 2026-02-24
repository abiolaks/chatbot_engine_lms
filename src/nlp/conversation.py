# src/nlp/conversation.py
import re
import logging
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class ConversationState:
    GREETING = "greeting"
    ASK_GOAL = "ask_goal"
    ASK_LEVEL = "ask_level"
    ASK_CAREER = "ask_career"
    RECOMMENDING = "recommending"
    ENDED = "ended"

class ConversationManager:
    def __init__(self):
        self.sessions = {}  # session_id -> state
    
    def new_session(self, session_id: str) -> Dict:
        self.sessions[session_id] = {
            "session_id": session_id,
            "state": ConversationState.GREETING,
            "info": {
                "goal": None,
                "level": None,
                "career": None
            },
            "history": [],
            "created_at": datetime.now().isoformat()
        }
        return self.sessions[session_id]
    
    def process_message(self, session_id: str, message: str) -> Dict:
        if session_id not in self.sessions:
            self.new_session(session_id)
        
        session = self.sessions[session_id]
        session["history"].append({"role": "user", "text": message, "time": datetime.now().isoformat()})
        
        # Extract info based on current state
        response_text, action, new_state = self._handle_state(session, message)
        
        session["state"] = new_state
        session["history"].append({"role": "bot", "text": response_text, "time": datetime.now().isoformat()})
        
        # If all info collected, prepare recommendation payload
        if new_state == ConversationState.RECOMMENDING:
            payload = self._build_recommendation_payload(session)
            return {
                "text": response_text,
                "action": "recommend",
                "collected_info": session["info"],
                "payload": payload
            }
        elif new_state == ConversationState.ENDED:
            return {
                "text": response_text,
                "action": "end",
                "collected_info": session["info"]
            }
        else:
            return {
                "text": response_text,
                "action": "continue",
                "collected_info": session["info"]
            }
    
    def _handle_state(self, session: Dict, message: str) -> tuple:
        state = session["state"]
        info = session["info"]
        msg_lower = message.lower()
        
        if state == ConversationState.GREETING:
            # Greet and ask for learning goal
            return ("Hi! I'm your learning assistant. What would you like to learn?",
                    "continue", ConversationState.ASK_GOAL)
        
        elif state == ConversationState.ASK_GOAL:
            # Extract goal
            goal = self._extract_goal(msg_lower)
            if goal:
                info["goal"] = goal
                return (f"Great! You want to learn {goal}. What's your experience level? (beginner, intermediate, advanced)",
                        "continue", ConversationState.ASK_LEVEL)
            else:
                # Could not extract, ask again
                return ("I didn't catch that. Could you tell me what skill you'd like to learn?",
                        "continue", ConversationState.ASK_GOAL)
        
        elif state == ConversationState.ASK_LEVEL:
            # Extract level
            level = self._extract_level(msg_lower)
            if level:
                info["level"] = level
                return ("Thanks. And what's your career goal or desired job role? (e.g., data scientist, web developer)",
                        "continue", ConversationState.ASK_CAREER)
            else:
                return ("Please specify your level: beginner, intermediate, or advanced.",
                        "continue", ConversationState.ASK_LEVEL)
        
        elif state == ConversationState.ASK_CAREER:
            # Extract career path
            career = self._extract_career(msg_lower)
            if career:
                info["career"] = career
                return ("Perfect! I have all the information. Let me find the best courses for you.",
                        "recommend", ConversationState.RECOMMENDING)
            else:
                # If no career given, we can still proceed
                info["career"] = "not specified"
                return ("No problem. Let me find courses based on your learning goal and level.",
                        "recommend", ConversationState.RECOMMENDING)
        
        elif state == ConversationState.RECOMMENDING:
            # After recommendations, ask if they want more help
            if "bye" in msg_lower or "thank" in msg_lower or "goodbye" in msg_lower:
                return ("You're welcome! Good luck with your learning. Feel free to come back anytime.",
                        "end", ConversationState.ENDED)
            else:
                return ("Would you like to explore other topics or say goodbye?",
                        "continue", ConversationState.RECOMMENDING)
        
        elif state == ConversationState.ENDED:
            return ("Goodbye!", "end", ConversationState.ENDED)
        
        else:
            return ("I'm not sure how to help. Can you rephrase?", "continue", state)
    
    def _extract_goal(self, text: str) -> Optional[str]:
        # Simple keyword matching; could be improved with a small model
        common_topics = ["python", "javascript", "data science", "machine learning", "web development",
                         "cloud", "devops", "ai", "design", "marketing", "excel", "sql"]
        for topic in common_topics:
            if topic in text:
                return topic
        # If no match, take the whole phrase but limit length
        words = text.split()
        if len(words) > 3:
            return " ".join(words[:3]) + "..."
        return text if text else None
    
    def _extract_level(self, text: str) -> Optional[str]:
        if "beginner" in text or "new" in text or "start" in text:
            return "beginner"
        if "intermediate" in text or "some experience" in text:
            return "intermediate"
        if "advanced" in text or "expert" in text:
            return "advanced"
        return None
    
    def _extract_career(self, text: str) -> Optional[str]:
        # Similar simple extraction
        careers = ["data scientist", "data analyst", "web developer", "software engineer",
                   "machine learning engineer", "devops engineer", "cloud architect"]
        for career in careers:
            if career in text:
                return career
        return text if text and len(text) < 50 else None
    
    def _build_recommendation_payload(self, session: Dict) -> Dict:
        return {
            "user_id": session["session_id"],
            "goal": session["info"]["goal"],
            "level": session["info"]["level"],
            "career": session["info"]["career"]
        }