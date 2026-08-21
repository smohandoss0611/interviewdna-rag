"""
Lightweight in-memory session store.

Holds the results of the pre-interview pipeline (resume text/DNA, job
text/DNA, alignment, strategy) between the /resume, /job, /match endpoints
and /interview/start, which seeds the LangGraph InterviewState from this data.

For a hackathon this in-memory dict is sufficient. Swap for Redis/a DB for
multi-worker deployments -- nothing else in the codebase depends on this
being in-memory.
"""
from __future__ import annotations

import uuid
from typing import Dict, Any, Optional


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str = "candidate") -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {"user_id": user_id}
        return session_id

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def update(self, session_id: str, **kwargs) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = {}
        self._sessions[session_id].update(kwargs)

    def require(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session


_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
