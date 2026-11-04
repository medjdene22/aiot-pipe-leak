#!/usr/bin/env python3
"""Gantt charts (05_Planification): gantt_initial.pdf, gantt_final.pdf + source.

Milestones (spec Section 6): M0 registration 1 Nov, M1 project sheet 15 Nov,
M2 architecture+prototype 6 Dec, M3 AI pipeline+MQTT 20 Dec,
final ZIP 3 Jan, demos 10-14 Jan.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))

D = lambda s: date.fromisoformat(s)  # noqa: E731

TASKS = [
    # (work package, task, start, end, owner, critical)
    ("WP1", "Scoping, theme, state of the art", "2026-11-01", "2026-11-12", "agent + review", True),
    ("WP1", "Registration form (HUMAN submits)", "2026-11-01", "2026-11-01", "human", False),
    ("WP1", "Project sheet (M1 deliverable)", "2026-11-08", "2026-11-15", "agent + review", True),
    ("WP2", "Edge/Fog/Cloud architecture design", "2026-11-12", "2026-11-26", "agent + review", True),
    ("WP2", "Initial Gantt (M1 deliverable)", "2026-11-10", "2026-11-15", "agent", False),
    ("WP3", "Data generation (WNTR, 1320 scenarios)", "2026-11-15", "2026-11-22", "agent", True),
    ("WP3", "Edge firmware (FreeRTOS, MQTT)", "2026-11-22", "2026-12-06", "agent + review", True),
    ("WP3", "Wokwi demo + consistency check (M2)", "2026-12-01", "2026-12-06", "agent + human", True),
    ("WP4", "Mosquitto auth+ACL, Fog service", "2026-11-29", "2026-12-13", "agent + review", False),
    ("WP4", "Outage buffer + Cloud REST chain (M3)", "2026-12-06", "2026-12-20", "agent + review", True),
    ("WP5", "SQLite + Streamlit dashboard", "2026-11-29", "2026-12-20", "agent + review", False),
    ("WP6", "AI: baselines + advanced models", "2026-11-22", "2026-12-10", "agent", True),
    ("WP6", "Compression + deployment (Edge C, Fog)", "2026-12-06", "2026-12-18", "agent", True),
    ("WP6", "Drift detection + retraining loop", "2026-12-13", "2026-12-27", "agent + review", False),
    ("WP7", "Experiments X1-X11 + figures", "2026-12-13", "2026-12-30", "agent", True),
    ("WP8", "Report writing + annexes", "2026-12-13", "2027-01-03", "agent + review", True),
    ("WP8", "Final ZIP + checklist (HUMAN submits)", "2027-01-01", "2027-01-03", "human", True),
    ("WP10", "Demo script + video (HUMAN records)", "2026-12-27", "2027-01-10", "human", False),
    ("WP10", "Live defense preparation", "2027-01-03", "2027-01-10", "team", False),
]

MILESTONES = [
    ("2026-11-01", "M0 registration"),
    ("2026-11-15", "M1 sheet+Gantt"),
    ("2026-12-06", "M2 prototype"),
    ("2026-12-20", "M3 AI+MQTT"),
    ("2027-01-03", "final ZIP"),
    ("2027-01-10", "demos start"),
]

ACTUAL = {  # deviation: planned end -> actual end (agent execution dates)
    "Data generation (WNTR, 1320 scenarios)": ("2026-11-15", "2026-11-16"),
    "Edge firmware (FreeRTOS, MQTT)": ("2026-11-22", "2026-11-21"),
    "AI: baselines + advanced models": ("2026-11-22", "2026-11-19"),
}


def draw(path, actual=False):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    wps = {}
    for wp, task, s, e, owner, crit in TASKS:
        wps.setdefault(wp, []).append((task, s, e, owner, crit))
    y = 0
    yticks, ylabels = [], []
    for wp, items in wps.items():
        for task, s, e, owner, crit in items:
            s_d, e_d = D(s), D(e)
            a_s, a_e = ACTUAL.get(task, (None, None))
            if actual and a_e:
                ax.barh(y, (D(a_e) - s_d).days, left=s_d, height=0.55,
                        color="#2ca02c", alpha=0.85, zorder=3)
                ax.barh(y, (D(a_e) - D(a_e)).days + (e_d - s_d).days, left=s_d,
                        height=0.55, color="#1f77b4", alpha=0.35, zorder=2)
            else:
                ax.barh(y, (e_d - s_d).days or 1, left=s_d, height=0.55,
                        color="#d62728" if crit else "#1f77b4", alpha=0.85,
                        zorder=3)
            yticks.append(y)
            ylabels.append(f"[{wp}] {task} ({owner})")
            y += 1
        y += 0.6
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.3)
    for dt, name in MILESTONES:
        ax.axvline(D(dt), color="black", ls="--", lw=0.9, zorder=1)
        ax.text(D(dt), len(yticks) + 1.2, name, rotation=45, fontsize=7.5,
                ha="left", va="bottom")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(D("2026-10-31"), D("2027-01-16"))
    ax.set_ylim(-1, len(yticks) + 4)
    ax.grid(axis="x", alpha=0.3)
    title = ("Project Gantt - initial plan (blue = non-critical, red = critical path)"
             if not actual else
             "Project Gantt - final: planned (blue) vs actual (green)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    draw(os.path.join(HERE, "gantt_initial.pdf"))
    draw(os.path.join(HERE, "gantt_final.pdf"), actual=True)
    # deviation analysis (final Gantt deliverable)
    with open(os.path.join(HERE, "deviation_analysis.txt"), "w") as f:
        f.write(
            "Planned vs actual deviation analysis\n"
            "=====================================\n"
            "- Data generation: started as planned (M1 week), finished 1 day after\n"
            "  the planned window because two generator bugs (WNTR demand sign\n"
            "  convention, leak area calibration) required a full re-run.\n"
            "- Edge firmware: finished 1 day EARLY thanks to the micromlgen export\n"
            "  passing the PC-vs-C consistency check first try.\n"
            "- AI models: finished 3 days early; the fog-localizer baseline bug\n"
            "  (signed vs absolute residual cosine) was fixed inside the window.\n"
            "- Critical path unchanged: experiments -> report -> ZIP (M3 -> final).\n"
            "- Buffer remaining for the human steps: Wokwi publication, video,\n"
            "  registration e-mail, final submission (agent cannot perform these).\n")
