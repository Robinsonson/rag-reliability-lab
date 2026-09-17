"""Run with python -m policy_lab.worker; leases recover after worker termination."""
import logging
import signal
import threading

from .config import Settings
from .repository import Repository
from .storage import make_storage


def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    repository = Repository(settings)
    repository.initialize()
    storage = make_storage(settings)
    storage.ensure_bucket()
    stop = threading.Event()
    def shutdown(*_):
        stop.set()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while not stop.is_set():
        try:
            if not repository.process_one_job(storage=storage):
                stop.wait(2)
        except Exception:
            logging.exception("Worker failed to claim/process a job")
            stop.wait(5)
    repository.engine.dispose()


if __name__ == "__main__":
    main()
