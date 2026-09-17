"""Plain-text output table."""
from __future__ import annotations


def format_table(result: dict) -> str:
    cols = ["Player", "Yards so far", "Line", "Games left", "Status", "Posterior mu median (80% int.)",
            "Proj. final median (10th-90th)", "P(hit)"]
    rows = []
    for r in result["players"]:
        rows.append([
            r["player"], f"{r['yards_so_far']:.0f}", f"{r['line']:.1f}", str(r["games_left"]), r["status"],
            f"{r['mu_median']:.1f} ({r['mu_p10']:.1f}-{r['mu_p90']:.1f})",
            f"{r['final_median']:.0f} ({r['final_p10']:.0f}-{r['final_p90']:.0f})",
            f"{100 * r['p_hit']:.1f}%",
        ])
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    line = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([line, sep] + body)


def format_summary(result: dict) -> str:
    be = 1.0 / result["payout_decimal"]
    return "\n".join([
        f"P(all 4 hit)          : {100 * result['p_all']:.2f}%   (product of individual P: {100 * result['p_product']:.2f}%;"
        f" break-even {100 * be:.2f}%)",
        f"Unit EV (per $1)      : {result['unit_ev']:+.4f}   (decimal payout {result['payout_decimal']:.6f})",
        f"EV of ${result['example_stake']:.0f} bet        : ${result['example_ev']:+,.2f}",
        f"Simulated seasons     : {result['n_sims']:,}  (seed {result['seed']})",
    ])


def format_result(result: dict) -> str:
    return format_table(result) + "\n\n" + format_summary(result)
