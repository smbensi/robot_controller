"""
Entity Resolver: resolves human-readable names to MongoDB document IDs.
Uses TTLCache to minimize database round-trips.

Modular design: the slot_type → collection mapping is driven entirely by
MongoSettings.collections. To support a new entity type, add one entry there —
no code changes needed here.
"""

from __future__ import annotations

import re
from typing import Optional

from cachetools import TTLCache
from pymongo import AsyncMongoClient

from src.config import AppSettings
from src.utils import get_logger

logger = get_logger(__name__)


class EntityNotFoundError(Exception):
    """Raised when a required entity slot cannot be resolved from MongoDB."""

    def __init__(self, slot_type: str, name: str, unknown_type: bool = False) -> None:
        self.slot_type = slot_type
        self.name = name
        self.unknown_type = unknown_type
        super().__init__(f"Entity not found: {slot_type}='{name}'")


class AmbiguousEntityError(Exception):
    """Raised when a name matches multiple documents and cannot be disambiguated."""

    def __init__(self, slot_type: str, name: str, matches: list[str]) -> None:
        self.slot_type = slot_type
        self.name = name
        self.matches = matches
        super().__init__(f"Ambiguous {slot_type}: '{name}' matches {matches}")


class EntityResolver:
    """
    Resolves entity names to MongoDB ObjectId strings.

    The mapping from command slot_type → MongoDB collection is configured in
    MongoSettings.collections. Add new entity types there — no code changes needed.

    Resolution strategy per slot_type:
    - "users"     → 3-tier name matching (see _resolve_user)
    - everything else → simple case-insensitive match on the "name" field
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings.mongo
        self._collections: dict[str, str] = settings.mongo.collections
        self._cache: TTLCache = TTLCache(
            maxsize=settings.entity_cache_maxsize,
            ttl=settings.entity_cache_ttl,
        )
        self._client: Optional[AsyncMongoClient] = None
        self._db = None

    async def connect(self) -> None:
        """Establish async MongoDB connection and ensure indexes on all collections."""
        self._client = AsyncMongoClient(self._settings.uri)
        self._db = self._client[self._settings.database]

        await self._client.admin.command("ping")
        logger.info(
            "mongodb_connected",
            uri=self._settings.uri,
            database=self._settings.database,
            collections=list(self._collections.values()),
        )

        # Indexes for users (firstname + lastname lookups)
        users_col = self._collections.get("users")
        if users_col:
            await self._db[users_col].create_index("firstname")
            await self._db[users_col].create_index("lastname")

        # Index for all other collections (name field lookup)
        for slot_type, collection in self._collections.items():
            if slot_type != "users":
                await self._db[collection].create_index("name")

    async def resolve_by_type(self, slot_type: str, name: str) -> str:
        """
        Resolve an entity name to its MongoDB _id for the given slot_type.

        Raises:
          EntityNotFoundError  — slot_type unknown, or name not found in DB
          AmbiguousEntityError — name matches multiple documents
        """
        collection = self._collections.get(slot_type)
        if collection is None:
            logger.warning(
                "unknown_slot_type",
                slot_type=slot_type,
                known_types=list(self._collections.keys()),
            )
            raise EntityNotFoundError(slot_type, name, unknown_type=True)

        cache_key = f"{collection}:{name.lower().strip()}"
        if cache_key in self._cache:
            logger.debug("entity_cache_hit", key=cache_key)
            return self._cache[cache_key]

        if self._db is None:
            logger.error("mongodb_not_connected")
            raise EntityNotFoundError(slot_type, name)

        if slot_type == "users":
            entity_id = await self._resolve_user(collection, name, slot_type)
        else:
            entity_id = await self._resolve_by_name(collection, name, slot_type)

        self._cache[cache_key] = entity_id
        return entity_id

    # ------------------------------------------------------------------ #
    #  Resolution strategies                                              #
    # ------------------------------------------------------------------ #

    async def _resolve_user(self, collection: str, name: str, slot_type: str) -> str:
        """
        3-tier user resolution:
          1. Full name — firstname+lastname or lastname+firstname (any word order)
          2. Firstname only
          3. Lastname only
        Raises AmbiguousEntityError when multiple users match at any tier.
        Raises EntityNotFoundError when no match is found at all.
        """
        name = name.strip()
        parts = name.split()

        # --- Tier 1: full name match (two words, either order) ---
        if len(parts) >= 2:
            first_word = re.compile(f"^{re.escape(parts[0])}$", re.IGNORECASE)
            last_word = re.compile(f"^{re.escape(parts[-1])}$", re.IGNORECASE)
            query = {"$or": [
                {"firstname": first_word, "lastname": last_word},
                {"firstname": last_word, "lastname": first_word},
            ]}
            docs = await self._db[collection].find(query).to_list(None)
            if len(docs) == 1:
                entity_id = str(docs[0]["_id"])
                logger.info("user_resolved_full_name", name=name, id=entity_id)
                return entity_id
            if len(docs) > 1:
                matches = [
                    f"{d.get('firstname', '')} {d.get('lastname', '')}".strip()
                    for d in docs
                ]
                raise AmbiguousEntityError(slot_type, name, matches)

        # --- Tier 2: firstname match ---
        pattern = re.compile(f"^{re.escape(name)}$", re.IGNORECASE)
        docs = await self._db[collection].find({"firstname": pattern}).to_list(None)
        if len(docs) == 1:
            entity_id = str(docs[0]["_id"])
            logger.info("user_resolved_firstname", name=name, id=entity_id)
            return entity_id
        if len(docs) > 1:
            matches = [
                f"{d.get('firstname', '')} {d.get('lastname', '')}".strip()
                for d in docs
            ]
            raise AmbiguousEntityError(slot_type, name, matches)

        # --- Tier 3: lastname match ---
        docs = await self._db[collection].find({"lastname": pattern}).to_list(None)
        if len(docs) == 1:
            entity_id = str(docs[0]["_id"])
            logger.info("user_resolved_lastname", name=name, id=entity_id)
            return entity_id
        if len(docs) > 1:
            matches = [
                f"{d.get('firstname', '')} {d.get('lastname', '')}".strip()
                for d in docs
            ]
            raise AmbiguousEntityError(slot_type, name, matches)

        logger.warning("user_not_found", name=name)
        raise EntityNotFoundError(slot_type, name)

    async def _resolve_by_name(self, collection: str, name: str, slot_type: str) -> str:
        """Simple case-insensitive match on the document's 'name' field."""
        pattern = re.compile(f"^{re.escape(name.strip())}$", re.IGNORECASE)
        doc = await self._db[collection].find_one({"name": pattern})
        if doc:
            entity_id = str(doc["_id"])
            logger.info("entity_resolved", collection=collection, name=name, id=entity_id)
            return entity_id
        logger.warning("entity_not_found", collection=collection, name=name)
        raise EntityNotFoundError(slot_type, name)

    async def health_check(self) -> bool:
        """Check MongoDB connectivity."""
        try:
            if self._client:
                await self._client.admin.command("ping")
                return True
        except Exception:
            pass
        return False

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("mongodb_disconnected")
