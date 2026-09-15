import json
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "interaction-events"


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    )


def publish_event(producer, event):
    payload = json.dumps(event).encode("utf-8")

    future = producer.send(KAFKA_TOPIC, value=payload)
    metadata = future.get(timeout=10)

    print(f"Topic: {metadata.topic}")
    print(f"Partition: {metadata.partition}")
    print(f"Offset: {metadata.offset}")


if __name__ == "__main__":
    event = {
        "event_id": str(uuid.uuid4()),
        "visitorid": 12345,
        "itemid": 67890,
        "event_type": "view",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": "session-demo-001",
        "context": {
            "device": "desktop",
            "query": "running shoes",
        },
    }

    producer = create_producer()

    try:
        publish_event(producer, event)
        print("Event published successfully")
    finally:
        producer.close()
