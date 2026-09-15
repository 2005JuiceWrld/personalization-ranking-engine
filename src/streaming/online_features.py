import redis


REDIS_HOST = "localhost"
REDIS_PORT = 6379


class OnlineFeatureStore:

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
    ):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
        )

    def get_recent_items(
        self,
        visitorid: int,
    ):
        key = f"user:{visitorid}:recent_items"

        return self.redis_client.lrange(
            key,
            0,
            -1,
        )

    def get_event_counts(
        self,
        visitorid: int,
    ):
        key = f"user:{visitorid}:event_counts"

        return self.redis_client.hgetall(
            key
        )

    def get_behavior_score(
        self,
        visitorid: int,
    ):
        key = f"user:{visitorid}:behavior_score"

        value = self.redis_client.get(
            key
        )

        if value is None:
            return 0

        return int(value)

    def get_last_event_timestamp(
        self,
        visitorid: int,
    ):
        key = f"user:{visitorid}:last_event_ts"

        return self.redis_client.get(
            key
        )

    def get_user_features(
        self,
        visitorid: int,
    ):
        return {
            "recent_items": self.get_recent_items(
                visitorid
            ),
            "event_counts": self.get_event_counts(
                visitorid
            ),
            "behavior_score": self.get_behavior_score(
                visitorid
            ),
            "last_event_ts": self.get_last_event_timestamp(
                visitorid
            ),
        }

    def close(self):
        self.redis_client.close()