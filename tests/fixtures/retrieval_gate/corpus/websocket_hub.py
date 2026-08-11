"""Broadcast hub tracking websocket subscribers per topic."""

import logging
import threading

logger = logging.getLogger(__name__)

MAX_SUBSCRIBERS_PER_TOPIC = 5000


class TopicFull(Exception):
    """Raised when a topic already holds its maximum subscriber count."""


class BroadcastHub:
    """Routes published messages to the sockets subscribed to a topic.

    Dead sockets are collected during the broadcast that discovers them rather
    than by a sweep, because a sweep either runs too often to be free or too
    rarely to keep the subscriber set honest.
    """

    def __init__(self, max_subscribers=MAX_SUBSCRIBERS_PER_TOPIC):
        self.max_subscribers = max_subscribers
        self._topics = {}
        self._lock = threading.Lock()

    def subscribe(self, topic, socket):
        """Add a socket to a topic's subscriber set."""
        with self._lock:
            subscribers = self._topics.setdefault(topic, set())
            if len(subscribers) >= self.max_subscribers:
                raise TopicFull(f"topic '{topic}' already has {len(subscribers)} subscribers")
            subscribers.add(socket)
            return len(subscribers)

    def unsubscribe(self, topic, socket):
        """Remove a socket from a topic, dropping the topic when it empties."""
        with self._lock:
            subscribers = self._topics.get(topic)
            if not subscribers:
                return 0
            subscribers.discard(socket)
            if not subscribers:
                del self._topics[topic]
            return len(subscribers)

    def broadcast(self, topic, message):
        """Send a message to every subscriber, dropping sockets that fail."""
        with self._lock:
            subscribers = list(self._topics.get(topic, ()))

        delivered, stale = 0, []
        for socket in subscribers:
            try:
                socket.send(message)
                delivered += 1
            except Exception as exc:
                logger.warning("Dropping stale subscriber on topic %s: %s", topic, exc)
                stale.append(socket)

        for socket in stale:
            self.unsubscribe(topic, socket)
        return delivered

    def topic_sizes(self):
        """Subscriber count per topic."""
        with self._lock:
            return {topic: len(subs) for topic, subs in self._topics.items()}
