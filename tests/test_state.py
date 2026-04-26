import sqlite3
import threading

from magic_claw.state import StateStore


def test_state_store_accepts_events_from_background_thread(tmp_path):
    db_path = tmp_path / "state.sqlite"
    state = StateStore(db_path)
    errors: list[BaseException] = []

    def worker():
        try:
            state.event("info", "telegram", "telegram polling started")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM events WHERE source = 'telegram'").fetchone()[0]
    assert count == 1
