"""Outbound email queue with per-message retry and a dead letter store."""

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)

MAX_DELIVERY_ATTEMPTS = 5
BASE_RETRY_DELAY_SECONDS = 2.0


class Undeliverable(Exception):
    """Raised when a message has exhausted its delivery attempts."""


class OutboundMessage:
    """One queued email, carrying its own attempt count."""

    def __init__(self, recipient, subject, body):
        self.recipient = recipient
        self.subject = subject
        self.body = body
        self.attempts = 0
        self.last_error = None

    def next_delay(self):
        """Seconds to wait before the next delivery attempt for this message."""
        return BASE_RETRY_DELAY_SECONDS * (2 ** (self.attempts - 1))


class EmailQueue:
    """A worker-backed queue that retries transient delivery failures.

    Retries live with the message rather than the worker so that a message
    requeued after a failure keeps its attempt count; a per-worker counter would
    reset every time the message moved to a different worker.
    """

    def __init__(self, transport, workers=2):
        self.transport = transport
        self._queue = queue.Queue()
        self._dead_letters = []
        self._lock = threading.Lock()
        self._workers = [threading.Thread(target=self._drain, daemon=True) for _ in range(workers)]

    def start(self):
        """Start the delivery workers."""
        for worker in self._workers:
            worker.start()

    def enqueue(self, recipient, subject, body):
        """Add a message to the outbound queue."""
        message = OutboundMessage(recipient, subject, body)
        self._queue.put(message)
        return message

    def _drain(self):
        """Deliver queued messages, requeueing on transient failure."""
        while True:
            message = self._queue.get()
            try:
                message.attempts += 1
                self.transport.send(message.recipient, message.subject, message.body)
            except Exception as exc:
                message.last_error = str(exc)
                if message.attempts >= MAX_DELIVERY_ATTEMPTS:
                    self._park(message)
                else:
                    time.sleep(message.next_delay())
                    self._queue.put(message)
            finally:
                self._queue.task_done()

    def _park(self, message):
        """Move an undeliverable message to the dead letter store."""
        logger.error("Giving up on message to %s after %d attempts", message.recipient, message.attempts)
        with self._lock:
            self._dead_letters.append(message)

    def dead_letters(self):
        """Messages that exhausted their delivery attempts."""
        with self._lock:
            return list(self._dead_letters)
