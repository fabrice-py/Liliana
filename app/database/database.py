"""Connexion SQLite et initialisation du schéma.

Une connexion par thread (``threading.local``) : FastAPI sert les requêtes dans
un pool de threads, et une connexion sqlite3 n'est pas partageable entre threads.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import DatabaseError
from app.core.logger import get_logger
from app.database.schema import SCHEMA_SQL, SCHEMA_VERSION

logger = get_logger(__name__)

_local = threading.local()
_init_lock = threading.Lock()
_initialised: set[str] = set()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=15.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def get_connection() -> sqlite3.Connection:
    """Connexion SQLite du thread courant, schéma garanti initialisé."""
    path = get_settings().database_path
    key = str(path)

    connection: sqlite3.Connection | None = getattr(_local, "connection", None)
    if connection is not None and getattr(_local, "path", None) == key:
        return connection

    if connection is not None:  # la config a changé (tests) : on repart proprement
        connection.close()

    try:
        connection = _connect(path)
    except sqlite3.Error as exc:  # pragma: no cover - dépend du système de fichiers
        raise DatabaseError(f"cannot open database at {path}: {exc}") from exc

    _local.connection = connection
    _local.path = key
    _ensure_schema(connection, key)
    return connection


def _ensure_schema(connection: sqlite3.Connection, key: str) -> None:
    with _init_lock:
        if key in _initialised:
            return
        try:
            connection.executescript(SCHEMA_SQL)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"cannot initialise schema: {exc}") from exc
        _initialised.add(key)
        logger.debug("Schéma SQLite v%s prêt (%s)", SCHEMA_VERSION, key)


def init_database() -> None:
    """Crée le fichier de base et le schéma. Appelé au démarrage."""
    get_connection()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Contexte transactionnel : commit à la sortie, rollback en cas d'erreur."""
    connection = get_connection()
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def reset_connection() -> None:
    """Ferme la connexion du thread courant (utilisé par les tests)."""
    connection: sqlite3.Connection | None = getattr(_local, "connection", None)
    if connection is not None:
        connection.close()
    _local.connection = None
    _local.path = None


def forget_initialised(path: str | Path | None = None) -> None:
    """Oublie l'état d'initialisation du schéma (utilisé par les tests)."""
    with _init_lock:
        if path is None:
            _initialised.clear()
        else:
            _initialised.discard(str(path))
