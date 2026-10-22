"""Container healthcheck for the background worker heartbeat."""

from __future__ import annotations

import socket

import psycopg

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    worker_prefix = f"{socket.gethostname()}:%"
    try:
        with psycopg.connect(settings.psycopg_conninfo, connect_timeout=3) as connection:
            fresh = connection.execute(
                """SELECT EXISTS (
                       SELECT 1
                         FROM worker_heartbeats
                        WHERE worker_id LIKE %s
                          AND last_seen > now() - interval '45 seconds'
                   )""",
                (worker_prefix,),
            ).fetchone()
    except psycopg.Error as exc:
        raise SystemExit(1) from exc
    raise SystemExit(0 if fresh and fresh[0] else 1)


if __name__ == "__main__":
    main()
