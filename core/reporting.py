# -*- coding: utf-8 -*-
"""Shared report fragments for PlanX GeoStats Lab HTML outputs."""
from __future__ import annotations

import html


def analyst_guidance_html(
    method_name: str,
    what_it_means: str,
    trust_conditions: list[str],
    red_flags: list[str],
    next_steps: list[str],
    planning_interpretation: str,
) -> str:
    """Return a standard interpretation block for professional analytical reports."""
    return (
        "<section class=\"advisor-block\">"
        f"<h2>{html.escape(method_name)} Analyst Guidance</h2>"
        "<div class=\"guidance-grid\">"
        f"{_guidance_card('What this means', [what_it_means])}"
        f"{_guidance_card('When to trust this', trust_conditions)}"
        f"{_guidance_card('Red flags', red_flags)}"
        f"{_guidance_card('Recommended next tools', next_steps)}"
        "</div>"
        f"<div class=\"note\"><strong>Planning interpretation:</strong> {html.escape(planning_interpretation)}</div>"
        "</section>"
    )


def analyst_guidance_css() -> str:
    """Return CSS used by shared analyst-guidance report blocks."""
    return """
.advisor-block { margin-top: 28px; }
.guidance-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 14px 0 20px; }
.guidance-card { background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; }
.guidance-card h3 { margin: 0 0 8px; color: #1f2937; font-size: .94rem; }
.guidance-card ul { margin: 0; padding-left: 18px; }
.guidance-card li { margin: 4px 0; }
"""


def _guidance_card(title: str, items: list[str]) -> str:
    body = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<div class=\"guidance-card\"><h3>{html.escape(title)}</h3><ul>{body}</ul></div>"
