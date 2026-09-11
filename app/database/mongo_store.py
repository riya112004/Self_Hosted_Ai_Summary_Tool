"""
MongoDB read-only store.

Only read operations are reachable:
  - find          (filter/projection/sort/limit)
  - aggregate     (pipeline, write + JS stages blocked)
  - count         (filter)
  - distinct      (field + filter)

Documents are normalized to plain JSON (ObjectId, Decimal128, datetime ->
str/float/iso) and the internal _id field is stripped.

The store refuses to connect to non-mongodb URI schemes.
"""
import json

from pymongo import MongoClient
from bson.json_util import dumps as bson_dumps

from .base import ReadOnlyStore, QueryRejected, guard_query, DEFAULT_MAX_ROWS

ALLOWED_URI_SCHEMES = ("mongodb://", "mongodb+srv://")


class MongoStore(ReadOnlyStore):
    def __init__(self, config: dict | None = None, max_rows: int = DEFAULT_MAX_ROWS):
        super().__init__(config)
        self.max_rows = max_rows
        self._client = None

    def connect(self) -> None:
        uri = self.config.get("uri") or "mongodb://localhost:27017"
        if not uri.startswith(ALLOWED_URI_SCHEMES):
            raise QueryRejected(
                f"Only mongodb/mongodb+srv URIs are allowed, got scheme of {uri!r}"
            )
        self._client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    def _collection(self):
        if self._client is None:
            raise QueryRejected("Store is not connected. Call connect() first.")
        return self._client[self.config.get("db") or "test"][
            self.config.get("collection") or "records"
        ]

    @staticmethod
    def _to_json(doc: dict) -> dict:
        clean = json.loads(bson_dumps(doc))
        clean.pop("_id", None)
        return clean

    def run(self, statement: dict) -> dict:
        limit = guard_query(statement, max_rows=self.max_rows)
        query_type = statement["query_type"]
        col = self._collection()

        if query_type == "find":
            filt = statement.get("query") or {}
            if not isinstance(filt, dict):
                raise QueryRejected("find query must be a dict filter")
            projection = {"_id": 0, **(statement.get("projection") or {})}
            cursor = col.find(filt, projection).limit(limit)
            sort = statement.get("sort")
            if sort:
                if not isinstance(sort, dict):
                    raise QueryRejected("sort must be a dict of field -> 1/-1")
                cursor = cursor.sort([(k, v) for k, v in sort.items()])
            records = [self._to_json(d) for d in cursor]

        elif query_type == "aggregate":
            pipeline = list(statement.get("query") or [])
            records = [
                self._to_json(d)
                for d in col.aggregate([*pipeline, {"$limit": limit}])
            ]

        elif query_type == "count":
            filt = statement.get("query") or {}
            if not isinstance(filt, dict):
                raise QueryRejected("count query must be a dict filter")
            n = col.count_documents(filt)
            return {"records": [], "count": n}

        else:  # distinct
            key = statement.get("key")
            if not isinstance(key, str) or not key:
                raise QueryRejected("distinct requires a string 'key'")
            filt = statement.get("query") or {}
            if not isinstance(filt, dict):
                raise QueryRejected("distinct query must be a dict filter")
            records = [{"value": v} for v in col.distinct(key, filt)]

        return {"records": records, "count": len(records)}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None