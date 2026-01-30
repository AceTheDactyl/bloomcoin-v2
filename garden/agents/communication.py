"""
Communication system for AI agents
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import time


@dataclass
class Message:
    """Message between agents"""
    sender_id: str
    sender_name: str
    recipient_id: str
    content: Dict[str, Any]
    message_type: str = "general"
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Dialogue:
    """Conversation between agents"""
    participants: List[str]
    messages: List[Message]
    topic: str = ""
    started_at: float = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = time.time()


class AgentCommunicator:
    """Handles agent communication"""

    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.sent_messages = []
        self.received_messages = []

    def send_message(self, recipient_id: str, content: Dict[str, Any], message_type: str = "general") -> Message:
        """Send a message to another agent"""
        message = Message(
            sender_id=self.agent_id,
            sender_name=self.agent_name,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type
        )
        self.sent_messages.append(message)
        return message

    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.received_messages.append(message)