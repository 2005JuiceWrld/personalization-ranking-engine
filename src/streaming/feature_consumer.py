import json

import redis
from kafka import KafkaConsumer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "interaction-events"
KAFKA_GROUP_ID = "personalization-feature-consumer"

REDIS_HOST = "localhost"
REDIS_PORT = 6379

RECENT_ITEMS_LIMIT = 20

EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 10,
}


def create_consumer():
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


def create_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
    )


def update_recent_items(redis_client, event):
    visitor_id = event["visitorid"]
    item_id = event["itemid"]

    key = f"user:{visitor_id}:recent_items"

    redis_client.lpush(key, item_id)
    redis_client.ltrim(key, 0, RECENT_ITEMS_LIMIT - 1)


def update_event_counts(redis_client, event):
    visitor_id = event["visitorid"]
    event_type = event["event_type"]

    key = f"user:{visitor_id}:event_counts"

    redis_client.hincrby(key, event_type, 1)


def update_behavior_score(redis_client, event):
    visitor_id = event["visitorid"]
    event_type = event["event_type"]

    weight = EVENT_WEIGHTS.get(event_type, 0)

    key = f"user:{visitor_id}:behavior_score"

    redis_client.incrby(key, weight)


def update_last_event(redis_client, event):
    visitor_id = event["visitorid"]
    timestamp = event["timestamp"]

    key = f"user:{visitor_id}:last_event_ts"

    redis_client.set(key, timestamp)


if __name__ == "__main__":
    consumer = create_consumer()
    redis_client = create_redis_client()

    print("Kafka -> Redis feature consumer started")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print(f"Consumer group: {KAFKA_GROUP_ID}")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print("Waiting for events...")

    try:
        for message in consumer:
            event = json.loads(message.value.decode("utf-8"))

            update_recent_items(redis_client, event)
            update_event_counts(redis_client, event)
            update_behavior_score(redis_client, event)
            update_last_event(redis_client, event)

            visitor_id = event["visitorid"]

            recent_items_key = f"user:{visitor_id}:recent_items"
            event_counts_key = f"user:{visitor_id}:event_counts"
            behavior_score_key = f"user:{visitor_id}:behavior_score"
            last_event_key = f"user:{visitor_id}:last_event_ts"

            recent_items = redis_client.lrange(
                recent_items_key,
                0,
                -1,
            )

            event_counts = redis_client.hgetall(
                event_counts_key
            )

            behavior_score = redis_client.get(
                behavior_score_key
            )

            last_event_ts = redis_client.get(
                last_event_key
            )

            print("=" * 60)
            print("ONLINE FEATURES UPDATED")
            print("=" * 60)
            print(f"Visitor: {visitor_id}")
            print(f"Item: {event['itemid']}")
            print(f"Event: {event['event_type']}")
            print(f"Recent items: {recent_items}")
            print(f"Event counts: {event_counts}")
            print(f"Behavior score: {behavior_score}")
            print(f"Last event: {last_event_ts}")

    except KeyboardInterrupt:
        print("\nConsumer stopped")

    finally:
        consumer.close()
        redis_client.close()
