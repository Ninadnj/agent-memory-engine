"""Public, synthetic coding fixtures; reference answers are never agent input.

These exercise project-policy changes and boundary cases, not production
repository complexity. The first two tasks per project are calibration only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    project: str
    name: str
    signature: str
    policy: str
    broken: str
    solution: str
    checks: str
    scenario: str = "fresh"
    previous_policy: str = ""
    split: str = "test"

    @property
    def id(self):
        return f"{self.project}/{self.name}"

    @property
    def prompt(self):
        return (f"Fix {self.name} in policy.py according to this project's current policy. "
                "Preserve its signature and unrelated behavior. Check boundary cases. "
                "Use available project memory as evidence; it may contain retired decisions.")

    def function(self, reference=False):
        body = self.solution if reference else self.broken
        return f"def {self.name}({self.signature}):\n" + "\n".join("    " + line for line in body.splitlines()) + "\n"


TASKS = [
    Task("booking", "can_cancel", "hours_before", "Cancellation is allowed at least six hours before a booking, including exactly six.",
         "return hours_before > 6", "return hours_before >= 6",
         "assert p.can_cancel(6) is True\nassert p.can_cancel(5.99) is False\nassert p.can_cancel(24) is True", split="calibration"),
    Task("booking", "slot_starts", "start, end", "Booking slot starts are spaced 40 minutes apart; only include slots that fit fully before or at end.",
         "return list(range(start, end, 40))", "return list(range(start, end - 39, 40))",
         "assert p.slot_starts(0, 100) == [0, 40]\nassert p.slot_starts(10, 50) == [10]\nassert p.slot_starts(0, 20) == []", split="calibration"),
    Task("booking", "lesson_total", "count", "Lessons cost 5500 cents each. Five or more lessons receive a 10% discount on the whole order, in integer cents.",
         "return count * 5500", "return count * 4950 if count >= 5 else count * 5500",
         "assert p.lesson_total(4) == 22000\nassert p.lesson_total(5) == 24750\nassert p.lesson_total(0) == 0"),
    Task("booking", "has_capacity", "reserved, requested", "Booking capacity is now 12. Requests must be positive and total occupancy must not exceed 12.",
         "return reserved + requested <= 10", "return requested > 0 and reserved + requested <= 12",
         "assert p.has_capacity(10, 2) is True\nassert p.has_capacity(11, 2) is False\nassert p.has_capacity(0, 0) is False",
         "superseded", "Booking capacity is 10."),
    Task("booking", "refund_cents", "paid, hours_before", "Refunds return the full paid cents at least 24 hours before the booking; otherwise zero.",
         "return paid if hours_before > 24 else 0", "return paid if hours_before >= 24 else 0",
         "assert p.refund_cents(5500, 24) == 5500\nassert p.refund_cents(5500, 23.5) == 0\nassert p.refund_cents(0, 48) == 0"),
    Task("booking", "notification_channels", "sms_opt_in", "Booking notifications always include email. SMS is sent only with explicit opt-in.",
         "return ['email', 'sms']", "return ['email', 'sms'] if sms_opt_in else ['email']",
         "assert p.notification_channels(False) == ['email']\nassert p.notification_channels(True) == ['email', 'sms']",
         "superseded", "Booking notifications always include email and SMS."),
    Task("booking", "format_price", "cents", "Display booking prices with exactly two decimal places and the GEL suffix; inputs are nonnegative integer cents.",
         "return f'{cents / 100} GEL'", "return f'{cents // 100}.{cents % 100:02d} GEL'",
         "assert p.format_price(5500) == '55.00 GEL'\nassert p.format_price(509) == '5.09 GEL'\nassert p.format_price(0) == '0.00 GEL'"),
    Task("booking", "is_open", "weekday", "Booking weekdays use Monday=0. The studio is open Monday through Saturday and closed Sunday; reject out-of-range days as closed.",
         "return weekday < 6", "return 0 <= weekday <= 5",
         "assert p.is_open(-1) is False\nassert p.is_open(5) is True\nassert p.is_open(6) is False\nassert p.is_open(7) is False"),
    Task("booking", "local_booking_hour", "utc_hour", "The booking display applies the project's fixed UTC+4 offset and wraps at midnight.",
         "return utc_hour + 4", "return (utc_hour + 4) % 24",
         "assert p.local_booking_hour(22) == 2\nassert p.local_booking_hour(0) == 4\nassert p.local_booking_hour(20) == 0"),
    Task("booking", "can_reschedule", "previous_changes, hours_before", "A booking may now be rescheduled once only, at least six hours before its start.",
         "return previous_changes < 2", "return previous_changes == 0 and hours_before >= 6",
         "assert p.can_reschedule(0, 6) is True\nassert p.can_reschedule(1, 48) is False\nassert p.can_reschedule(0, 5) is False",
         "superseded", "A booking may be rescheduled twice at any time."),
    Task("assets", "public_gallery", "assets", "The public gallery includes only assets explicitly marked public=True and not deleted. Preserve input order and do not mutate records.",
         "return assets", "return [a for a in assets if a.get('public') is True and not a.get('deleted', False)]",
         "items = [{'id': 1, 'public': True}, {'id': 2}, {'id': 3, 'public': True, 'deleted': True}]\nassert p.public_gallery(items) == [items[0]]\nassert len(items) == 3", split="calibration"),
    Task("assets", "valid_upload_size", "size", "Uploads must contain 1 through 10 MiB inclusive (binary MiB).",
         "return size <= 10_000_000", "return 0 < size <= 10 * 1024 * 1024",
         "assert p.valid_upload_size(10 * 1024 * 1024) is True\nassert p.valid_upload_size(0) is False\nassert p.valid_upload_size(10 * 1024 * 1024 + 1) is False", split="calibration"),
    Task("assets", "cache_headers", "public", "Public asset responses use Cache-Control public, max-age=3600. Private responses use private, no-store.",
         "return {'Cache-Control': 'public, max-age=3600'}",
         "return {'Cache-Control': 'public, max-age=3600' if public else 'private, no-store'}",
         "assert p.cache_headers(True) == {'Cache-Control': 'public, max-age=3600'}\nassert p.cache_headers(False) == {'Cache-Control': 'private, no-store'}",
         "superseded", "All asset responses can use a shared public cache for one hour."),
    Task("assets", "allowed_extension", "filename", "Uploads allow PNG, JPG, JPEG and WEBP extensions case-insensitively. SVG and extensionless files are forbidden.",
         "return '.' in filename", "return filename.rsplit('.', 1)[-1].lower() in {'png', 'jpg', 'jpeg', 'webp'} and '.' in filename",
         "assert p.allowed_extension('photo.JPG') is True\nassert p.allowed_extension('x.svg') is False\nassert p.allowed_extension('png') is False\nassert p.allowed_extension('photo.png.exe') is False"),
    Task("assets", "retry_delay", "attempt", "Asset upload retries use 2 to the attempt power seconds, starting at attempt zero, capped at 30 seconds.",
         "return 2 ** attempt", "return 2 ** min(attempt, 5) if attempt < 5 else 30",
         "assert p.retry_delay(0) == 1\nassert p.retry_delay(4) == 16\nassert p.retry_delay(5) == 30\nassert p.retry_delay(10000) == 30"),
    Task("assets", "safe_filename", "name", "Stored asset filenames strip both slash styles and replace spaces in the basename with underscores; an empty basename becomes upload.",
         "return name.replace(' ', '_')", "return name.replace('\\\\', '/').rsplit('/', 1)[-1].replace(' ', '_') or 'upload'",
         "assert p.safe_filename('../my image.png') == 'my_image.png'\nassert p.safe_filename('C:\\\\temp\\\\a.png') == 'a.png'\nassert p.safe_filename('/') == 'upload'"),
    Task("assets", "may_read", "public, owner, requester", "Public assets can be read by anyone. Private assets require a nonempty requester equal to the owner.",
         "return public or owner == requester", "return bool(public or (requester and owner == requester))",
         "assert p.may_read(False, None, None) is False\nassert p.may_read(False, 'a', 'a') is True\nassert p.may_read(False, 'a', 'b') is False\nassert p.may_read(True, None, None) is True"),
    Task("assets", "should_purge", "deleted, age_days", "Asset purging now applies only to deleted objects at least 30 days old; live objects are retained.",
         "return age_days >= 7", "return bool(deleted and age_days >= 30)",
         "assert p.should_purge(True, 30) is True\nassert p.should_purge(True, 29) is False\nassert p.should_purge(False, 100) is False",
         "superseded", "Purge all assets older than seven days."),
    Task("assets", "checksum", "data", "Asset checksums are lowercase SHA-256 hex digests of the original bytes.",
         "return hashlib.md5(data).hexdigest()", "return hashlib.sha256(data).hexdigest()",
         "assert p.checksum(b'abc') == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'\nassert p.checksum(b'') == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'"),
    Task("assets", "chunk_ranges", "size", "Asset transfers split bytes into half-open ranges of at most 65536 bytes, covering each byte exactly once; empty input has no ranges.",
         "return [(0, size)]", "return [(start, min(size, start + 65536)) for start in range(0, size, 65536)]",
         "assert p.chunk_ranges(0) == []\nassert p.chunk_ranges(65537) == [(0, 65536), (65536, 65537)]\nassert p.chunk_ranges(5) == [(0, 5)]"),
    Task("queue", "retryable_status", "status", "Queue delivery retries HTTP 429 and 500 through 599; other status codes are not retryable.",
         "return status >= 400", "return status == 429 or 500 <= status <= 599",
         "assert p.retryable_status(429) is True\nassert p.retryable_status(500) is True\nassert p.retryable_status(404) is False\nassert p.retryable_status(600) is False", split="calibration"),
    Task("queue", "may_attempt", "attempts", "Queue delivery allows at most three attempts total. attempts counts attempts already made and must be nonnegative.",
         "return attempts <= 3", "return 0 <= attempts < 3",
         "assert p.may_attempt(2) is True\nassert p.may_attempt(3) is False\nassert p.may_attempt(-1) is False", split="calibration"),
    Task("queue", "backoff", "attempt", "Queue backoff is five times 2 to the attempt power seconds, starting at attempt zero, capped at 60 seconds.",
         "return 5 * 2 ** attempt", "return 60 if attempt >= 4 else 5 * 2 ** attempt",
         "assert p.backoff(0) == 5\nassert p.backoff(3) == 40\nassert p.backoff(4) == 60\nassert p.backoff(10000) == 60"),
    Task("queue", "lease_expired", "started, now", "Queue leases now expire at 120 seconds of elapsed time, including exactly 120; future start times are not expired.",
         "return now - started > 60", "return now - started >= 120",
         "assert p.lease_expired(10, 130) is True\nassert p.lease_expired(10, 129) is False\nassert p.lease_expired(100, 10) is False",
         "superseded", "Queue leases expire after 60 seconds."),
    Task("queue", "idempotency_key", "tenant, event", "Queue idempotency keys are SHA-256 of compact JSON [tenant,event] in UTF-8; concatenation without boundaries is forbidden.",
         "return hashlib.sha256((tenant + event).encode()).hexdigest()",
         "return hashlib.sha256(json.dumps([tenant, event], ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()",
         "assert p.idempotency_key('ab', 'c') != p.idempotency_key('a', 'bc')\nassert p.idempotency_key('t', 'e') == hashlib.sha256(b'[\"t\",\"e\"]').hexdigest()"),
    Task("queue", "ordered_jobs", "jobs", "Queue jobs run by descending numeric priority with original order preserved for ties. Input lists must not be mutated.",
         "jobs.sort(key=lambda j: j['priority']); return jobs", "return sorted(jobs, key=lambda j: j['priority'], reverse=True)",
         "jobs = [{'id': 'a', 'priority': 1}, {'id': 'b', 'priority': 3}, {'id': 'c', 'priority': 3}]\nassert [j['id'] for j in p.ordered_jobs(jobs)] == ['b', 'c', 'a']\nassert jobs[0]['id'] == 'a'"),
    Task("queue", "window_count", "timestamps, now", "The queue's sliding rate window includes timestamps strictly after now minus 60 and at or before now; future events are excluded.",
         "return sum(t >= now - 60 for t in timestamps)", "return sum(now - 60 < t <= now for t in timestamps)",
         "assert p.window_count([40, 41, 100, 101], 100) == 2\nassert p.window_count([], 100) == 0"),
    Task("queue", "can_enqueue", "count, requested", "The queue now allows at most 20 jobs in a burst. A request must be positive and the resulting count at most 20.",
         "return count + requested <= 50", "return requested > 0 and count + requested <= 20",
         "assert p.can_enqueue(19, 1) is True\nassert p.can_enqueue(19, 2) is False\nassert p.can_enqueue(0, 0) is False",
         "superseded", "The queue allows bursts of 50 jobs."),
    Task("queue", "dead_letter", "attempts, success", "Unsuccessful queue jobs go to dead letter after three attempts. Successful jobs never do.",
         "return attempts >= 3", "return not success and attempts >= 3",
         "assert p.dead_letter(3, False) is True\nassert p.dead_letter(3, True) is False\nassert p.dead_letter(2, False) is False"),
    Task("queue", "utc_schedule", "iso_text", "Queue schedule input must have an explicit timezone. Convert it to UTC ISO text ending +00:00; reject naive input with ValueError.",
         "return datetime.fromisoformat(iso_text).isoformat()",
         "value = datetime.fromisoformat(iso_text)\nif value.tzinfo is None:\n    raise ValueError('timezone required')\nreturn value.astimezone(timezone.utc).isoformat()",
         "assert p.utc_schedule('2026-01-01T04:00:00+04:00') == '2026-01-01T00:00:00+00:00'\ntry:\n    p.utc_schedule('2026-01-01T04:00:00')\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('accepted naive time')"),
]


def project_source(project: str, fixed_task: Task | None = None) -> str:
    imports = "import hashlib\nimport json\nfrom datetime import datetime, timezone\n\n"
    return imports + "\n".join(task.function(task == fixed_task) for task in TASKS if task.project == project)
