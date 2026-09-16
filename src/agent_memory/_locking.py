"""OS-owned local file locks. Process exit releases the lock automatically."""

import os
import time
from contextlib import contextmanager
from pathlib import Path


def _replace_file(source: Path, target: Path, timeout: float = 1.0) -> None:
    """Allow a Windows reader to close before retrying an atomic replace.

    Writers still hold the store lock throughout. Windows denies replacement
    while a reader has the destination open; retries never unlink the old file
    and stop on a deadline. Other permission failures propagate immediately.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError as exc:
            if (
                getattr(exc, "winerror", None) not in (5, 32)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))


@contextmanager
def _file_lock(target: Path, timeout: float = 10.0, stale_after: float = 60.0):
    """Lock a persistent guard file, never unlinking another owner's inode.

    ``stale_after`` remains accepted for old Python callers but is unused:
    age is not evidence that the owning process has died. All writers sharing
    a store must use this protocol; v0.3's create/unlink locks are incompatible.
    Local filesystems only; network filesystem locking is outside the contract.
    """
    guard = target.with_name(target.name + ".guard")
    guard.parent.mkdir(parents=True, exist_ok=True)
    with guard.open("a+b") as stream:
        if guard.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        if os.name == "nt":
            import msvcrt

            def acquire():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except OSError as exc:
                import errno

                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not lock {target} after {timeout}s"
                    ) from exc
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            release()
