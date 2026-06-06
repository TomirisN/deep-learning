from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    topic: str
    objective: str
    step_id: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    status: str = "running"
    stop_reason: str = ""
