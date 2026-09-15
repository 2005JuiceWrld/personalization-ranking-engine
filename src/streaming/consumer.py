import json

from kafka import KafkaConsumer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "interaction-events"
KAFKA_GROUP_ID = "personalization-event-consumer"


def create_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )


if __name__ == "__main__":
    consumer = create_consumer()

    print("Kafka consumer started")
    print(f"Topic: {KAFKA_TOPIC}")
    print(f"Group: {KAFKA_GROUP_ID}")
    print("Waiting for events...")

    try:
        for message in consumer:
            event = message.value

            print("=" * 60)
            print("EVENT RECEIVED")
            print("=" * 60)
            print(json.dumps(event, indent=2))
            print(f"Partition: {message.partition}")
            print(f"Offset: {message.offset}")

    except KeyboardInterrupt:
        print("\nConsumer stopped")

    finally:
        consumer.close()
