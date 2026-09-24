"""When the server should top itself up.

A `serve` process is often the only thing running for days at a time, and the
index quietly goes stale behind it -- the reason "Fetch new papers" exists at
all. This module holds the rule for when to press that button automatically:
the stored setting, and the arithmetic deciding whether a run is due.

Nothing here runs anything or touches a thread. It is pure functions over a
config dict and two timestamps, which is what makes the awkward part -- "is a
run due, given it last ran then?" -- something that can be checked directly
rather than by waiting around for a scheduler.

The setting lives in the settings file beside the profile: it is a choice
made by whoever runs the server. When a run last started is about the index
instead, so that stays in papers.db.

Two modes, which together are the frequency and the timing:

    interval   every N hours, from when the last run started
    daily      at a wall-clock time, local to the machine running the server

Local time, not UTC, and deliberately: someone asking for 07:00 means 07:00
where they are, and arXiv's own announcement schedule is a fixed local time in
New York rather than anything the reader would compute in UTC.

Both modes catch up rather than skip. A daily 07:00 run on a server that was
asleep until 09:00 fires at 09:00, because the useful reading of "daily at
07:00" is "once a day, in the morning", not "only ever at exactly 07:00".
"""

import datetime as dt

from . import settings, store

KEY = "auto_update"
LAST_KEY = "auto_update_last"

MODES = ("off", "interval", "daily")

DEFAULT_MODE = "off"
DEFAULT_HOURS = 6.0
DEFAULT_AT = "07:00"

# One hour is already more often than the arXiv listing changes; the ceiling is
# a week, past which "automatic" stops meaning anything useful.
MIN_HOURS = 1.0
MAX_HOURS = 168.0


def clean(raw) -> dict:
    """Normalise a submitted setting. Anything unreadable falls back.

    Values are kept even when the mode does not use them, so switching from
    daily back to interval does not lose the hours that were set before.
    """
    if not isinstance(raw, dict):
        raw = {}

    mode = raw.get("mode")
    if mode not in MODES:
        mode = DEFAULT_MODE

    try:
        hours = float(raw.get("hours", DEFAULT_HOURS))
    except (TypeError, ValueError):
        hours = DEFAULT_HOURS
    if hours != hours:      # NaN
        hours = DEFAULT_HOURS
    hours = round(max(MIN_HOURS, min(MAX_HOURS, hours)), 2)

    return {"mode": mode, "hours": hours, "at": _clean_at(raw.get("at"))}


def _clean_at(raw) -> str:
    """A "HH:MM" wall-clock time, or the default if it is not one."""
    try:
        parsed = dt.datetime.strptime(str(raw).strip(), "%H:%M")
    except (TypeError, ValueError):
        return DEFAULT_AT
    return parsed.strftime("%H:%M")


def load() -> dict:
    return clean(settings.get(KEY))


def save(raw) -> dict:
    setting = clean(raw)
    settings.update(**{KEY: setting})
    return setting


def last_run(db) -> float:
    """When a run last started, as epoch seconds. 0 if none ever has."""
    try:
        return float(store.get_meta(db, LAST_KEY, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def note_run(db, when: float) -> None:
    """Record that a run started, so the next one is timed from it.

    Manual runs count. Someone who has just pressed "Fetch new papers" does not
    want the scheduler to do it again ten minutes later, and the record has to
    outlive the process or a restart would lose that.
    """
    store.set_meta(db, LAST_KEY, repr(float(when)))


def _at_on(day: dt.date, at: str) -> float:
    hour, minute = (int(part) for part in at.split(":"))
    return dt.datetime.combine(day, dt.time(hour, minute)).timestamp()


def next_run(setting, last: float, now: float):
    """When the next run is due, as epoch seconds, or None if never.

    A returned time in the past means overdue: the caller should run, and the
    UI should say so rather than printing a moment that has already gone.
    """
    if setting["mode"] == "interval":
        # With nothing recorded the first tick is due immediately, which is
        # what makes "every 6 hours" start working the moment it is switched
        # on rather than six hours later.
        return (last + setting["hours"] * 3600) if last else now

    if setting["mode"] == "daily":
        today = _at_on(dt.date.fromtimestamp(now), setting["at"])
        if now < today:
            # Not yet time today. Yesterday's slot still counts as missed if
            # nothing has run since it.
            yesterday = _at_on(
                dt.date.fromtimestamp(now) - dt.timedelta(days=1),
                setting["at"])
            return yesterday if last < yesterday else today
        return today if last < today else _at_on(
            dt.date.fromtimestamp(now) + dt.timedelta(days=1), setting["at"])

    return None


def due(setting, last: float, now: float) -> bool:
    """Whether a run should start now."""
    when = next_run(setting, last, now)
    return when is not None and now >= when
