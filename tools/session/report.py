"""Practice history from the session log (Methodology Stage 4/6, analytics).

Text version of the progress dashboard: what was practised, how long it was
held, which joint is the recurring weakness.  It reads only the derived numbers
in data/sessions.db - there is no video to read.

    python tools/session/report.py
    python tools/session/report.py --limit 20
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yoga.storage import DEFAULT_DB                             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Practice history report")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.db):
        print("No practice log yet at " + args.db)
        print("Run app.py once and the log is created automatically.")
        return 0

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    sessions = conn.execute(
        "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (args.limit,)).fetchall()
    if not sessions:
        print("No sessions recorded yet.")
        return 0

    print("")
    print("PRACTICE HISTORY  (" + args.db + ")")
    print("")
    head = ("id".rjust(4) + "  " + "started".ljust(26) + "asana".ljust(14)
            + "held".rjust(8) + "longest".rjust(9) + "done".rjust(6)
            + "best".rjust(8) + "fps".rjust(7))
    print(head)
    print("-" * len(head))
    for s in sessions:
        print(f"{s['id']:>4}  "
              + str(s["started_at"])[:25].ljust(26)
              + str(s["asana"])[:13].ljust(14)
              + f"{(s['total_hold_s'] or 0):7.1f}s"
              + f"{(s['longest_hold_s'] or 0):8.1f}s"
              + f"{(s['completed_holds'] or 0):6}"
              + f"{(s['best_score'] or 0):7.1f}%"
              + f"{(s['avg_fps'] or 0):7.1f}")

    totals = conn.execute(
        "SELECT COUNT(*) n, SUM(total_hold_s) held, SUM(completed_holds) done,"
        " MAX(best_score) best FROM sessions WHERE ended_at IS NOT NULL").fetchone()
    print("")
    print(f"all time: {totals['n'] or 0} sessions, "
          f"{(totals['held'] or 0) / 60.0:.1f} minutes in the pose, "
          f"{totals['done'] or 0} full holds, best alignment {(totals['best'] or 0):.1f}%")

    weak = conn.execute(
        "SELECT joint, COUNT(*) n, SUM(1 - in_tolerance) out_of_tol,"
        "       AVG(ABS(deviation)) mean_dev"
        "  FROM joint_samples GROUP BY joint HAVING n > 0"
        "  ORDER BY (1.0 * SUM(1 - in_tolerance) / n) DESC").fetchall()
    if weak:
        print("")
        print("RECURRING WEAKNESSES  (share of scored frames outside tolerance)")
        print("check".ljust(24) + "out of tol".rjust(12) + "mean |dev|".rjust(12))
        for w in weak:
            share = 100.0 * (w["out_of_tol"] or 0) / max(1, w["n"])
            print(w["joint"].ljust(24) + f"{share:11.0f}%" + f"{(w['mean_dev'] or 0):12.2f}")

    cues = conn.execute(
        "SELECT text, COUNT(*) n FROM cues WHERE joint NOT LIKE 'state.%'"
        " GROUP BY text ORDER BY n DESC LIMIT 5").fetchall()
    if cues:
        print("")
        print("MOST FREQUENT CORRECTIONS")
        for c in cues:
            print(f"  {c['n']:>4}x  {c['text']}")

    conn.close()
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
