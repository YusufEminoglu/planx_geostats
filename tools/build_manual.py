# -*- coding: utf-8 -*-
"""Build the reference manual from its single source.

``docs/GEOSTATS_REFERENCE_MANUAL.html`` is the manual's source of truth.  The
algorithm cards added from v3.11.0 on are written as data in
``tools/manual_src/*.py`` and rendered here into the same markup as the
hand-written cards, so a card cannot drift from the house format and a
sidebar entry cannot be forgotten.  The build also derives every number the
manual states about itself instead of trusting a hand-edited copy:

* the version in the title, sidebar, hero chip and footer - from metadata.txt
* the algorithm count in the hero chip - from the cards in the file
* the per-group tool counts in the Analytical Workflow strip - from the cards
  under each group header
* the root-level copy ``GEOSTATS_REFERENCE_MANUAL.html`` - byte-identical to
  docs/, written by this script and by nothing else

    py -3 tools/build_manual.py           # rebuild both copies
    py -3 tools/build_manual.py --check   # exit 1 if a rebuild would change anything

The script is development tooling (stdlib only) and is not shipped in the zip.
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The public manual lives in the portal repository (geophilo_project), not
# in this plugin repository or its Hub zip. GEOPHILO_MANUALS overrides it.
MANUALS = Path(os.environ.get("GEOPHILO_MANUALS") or ROOT.parents[1] / "geophilo_project" / "_manual_sources") / "planx_geostats"
MANUAL = MANUALS / "GEOSTATS_REFERENCE_MANUAL.html"
SOURCES = ROOT / "tools" / "manual_src"

P_STYLE = 'style="font-size:.84rem;color:var(--gs-slate-700);line-height:1.7;margin:8px 0 14px;"'


def metadata_version() -> str:
    for line in (ROOT / "metadata.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("version="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("metadata.txt has no version line")


def load_cards() -> list[dict]:
    cards = []
    for path in sorted(SOURCES.glob("cards_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cards.extend(getattr(module, "CARDS", []))
    return cards


def load_groups() -> list[dict]:
    groups = []
    for path in sorted(SOURCES.glob("cards_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem + "_g", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        groups.extend(getattr(module, "GROUPS", []))
    return groups


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _table(headers, rows, name_cols=1, type_col=None, default_col=None):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for row in rows:
        cells = []
        for k, value in enumerate(row):
            cls = ""
            if k < name_cols:
                cls = ' class="param-name"'
            elif type_col is not None and k == type_col:
                cls = ' class="param-type"'
            elif default_col is not None and k == default_col:
                cls = ' class="param-default"'
            cells.append(f"<td{cls}>{value}</td>")
        body.append("      <tr>" + "".join(cells) + "</tr>")
    return (f'  <table class="params-table">\n    <thead><tr>{head}</tr></thead>\n    <tbody>\n'
            + "\n".join(body) + "\n    </tbody>\n  </table>")


def render_card(card: dict, titles: dict) -> str:
    cid = card["id"]
    out = [f"<!-- {card['title']} -->",
           f'<article class="alg-card" id="{cid}">',
           '<div class="alg-card-header">',
           f'  <div class="alg-icon">{card.get("icon", "📊")}</div>',
           '  <div class="alg-info">',
           f"    <h4>{card['title']}</h4>",
           f'    <div class="alg-id">{card["alg"]}</div>',
           "  </div>",
           f'  <button class="copy-anchor-btn" data-anchor="{cid}" title="Copy link to this section" type="button">🔗</button>',
           "</div>",
           '<div class="alg-card-body">',
           '  <div class="alg-desc">',
           f"    {card['desc']}",
           "  </div>",
           "",
           "  <h5>1. Theoretical Background</h5>"]
    for title, text in card["background"]:
        out.append(f"  <p {P_STYLE}>")
        out.append(f"  <strong>{title}</strong> {text}")
        out.append("  </p>")
    out += ["", "  <h5>2. Mathematical Formulation</h5>"]
    if card.get("notation"):
        out.append(f'  <p style="font-size:.84rem;color:var(--gs-slate-600);margin:4px 0 10px;">{card["notation"]}</p>')
    out.append('  <div style="font-size:.88rem;line-height:1.6;margin:12px 0;padding:0 8px;">')
    for n, item in enumerate(card["formulas"], start=1):
        label, tex = item[0], item[1]
        out.append(f'  <p style="margin:8px 0;"><strong>{label}</strong></p>')
        out.append(f"  $${tex} \\tag{{{n}}}$$")
        if len(item) > 2 and item[2]:
            out.append(f'  <p style="margin:6px 0;">{item[2]}</p>')
    out.append("  </div>")
    if card.get("note"):
        out += ["", '  <div class="acad-note">', f"  <strong>Implementation note.</strong> {card['note']}", "  </div>"]
    out += ["", "  <h5>3. Parameters</h5>",
            _table(["Parameter", "Type", "Default", "Constraints", "Description"], card["params"],
                   type_col=1, default_col=2)]
    out += ["", "  <h5>4. Output Description</h5>"]
    if card.get("outputs"):
        out += ['  <div class="output-fields">',
                _table(["Field", "Type", "Description"], card["outputs"], type_col=1),
                "  </div>"]
    if card.get("report"):
        out.append(f'  <p style="font-size:.82rem;color:var(--gs-slate-700);margin-top:8px;"><strong>Report includes:</strong> {card["report"]}</p>')
    out += ["", "  <h5>5. Interpretation Guide</h5>", '  <div class="interp-box">',
            f"  <strong>{card['interp_title']}</strong>",
            '  <table class="params-table" style="margin:8px 0;">',
            "    <thead><tr><th>Signal</th><th>Meaning</th><th>Recommended Next Step</th></tr></thead>",
            "    <tbody>"]
    for signal, meaning, step in card["interp"]:
        out.append(f"      <tr><td>{signal}</td><td>{meaning}</td><td>{step}</td></tr>")
    out += ["    </tbody>", "  </table>", "  </div>"]
    for extra in card.get("interp_notes", []):
        out.append(f'  <div class="interp-box">{extra}</div>')
    if card.get("see_also"):
        links = " &middot; ".join(f'<a href="#{a}">{titles.get(a, a)}</a>' for a in card["see_also"])
        out += ["", '    <div class="see-also-box">', f"    <strong>See also:</strong> {links}", "  </div>"]
    out += ["", "<h5>6. Literature</h5>",
            '  <div style="font-size:.78rem;color:var(--gs-slate-500);line-height:1.55;margin-top:8px;">']
    for ref in card["refs"]:
        out.append(f'  <p style="margin:4px 0;">{ref}</p>')
    out += ["  </div>", "</div>", "</article>"]
    return "\n".join(out)


# --------------------------------------------------------------------------
# document surgery
# --------------------------------------------------------------------------

def _card_span(text: str, cid: str):
    """(start, end) of a card including the comment line directly above it."""
    start = text.find(f'<article class="alg-card" id="{cid}"')
    if start < 0:
        return None
    end = text.index("</article>", start) + len("</article>")
    line_start = text.rfind("\n", 0, start) + 1
    prev_start = text.rfind("\n", 0, line_start - 1) + 1
    prev_line = text[prev_start:line_start].strip()
    if prev_line.startswith("<!--") and prev_line.endswith("-->"):
        start = prev_start
    else:
        start = line_start
    return start, end


def sidebar_titles(text: str) -> dict:
    nav = text[text.find('<nav id="sidebar">'):text.find("</nav>")]
    return {m.group(1): m.group(2) for m in re.finditer(r'<a href="#([^"]+)">([^<]+)</a>', nav)}


def place_group(text: str, group: dict) -> str:
    """Insert a new group header (and its sidebar label) if it is missing."""
    marker = f'<span class="group-num">{group["num"]}</span>'
    if marker not in text:
        span = _card_span(text, group["after"])
        block = ("\n\n<!-- ════════════════════════════════════════════════════════════════ -->\n"
                 f"<!--  GROUP {group['num']} — {group['title'].upper():<57}-->\n"
                 "<!-- ════════════════════════════════════════════════════════════════ -->\n"
                 '<div class="group-header">\n'
                 f'  <span class="group-num">{group["num"]}</span>\n'
                 f"  <h3>{group['title']}</h3>\n</div>\n"
                 '<p style="color:var(--gs-slate-600);font-size:.88rem;margin-bottom:28px;">\n'
                 f"{group['intro']}\n</p>")
        text = text[:span[1]] + block + text[span[1]:]
    label = f'<div class="group-label">{group["num"]} — {group["sidebar"]}</div>'
    if label not in text:
        nav_anchor = '  <div class="group-label">Appendices</div>'
        text = text.replace(nav_anchor, f"  {label}\n\n{nav_anchor}", 1)
    step = f'<div class="step-num">{group["num"]}</div>'
    if step not in text:
        flow_end = text.index("</div>\n<p", text.index('<div class="workflow-flow">'))
        block = ('  <div class="workflow-arrow">→</div>\n  <div class="workflow-step">\n'
                 f'    <div class="step-num">{group["num"]}</div>\n'
                 f'    <div class="step-name">{group["step_name"]}</div>\n'
                 '    <div class="step-tools">0 tools</div>\n  </div>\n')
        text = text[:flow_end] + block + text[flow_end:]
    return text


def place_card(text: str, card: dict, titles: dict) -> str:
    rendered = render_card(card, titles)
    span = _card_span(text, card["id"])
    if span:
        return text[:span[0]] + rendered + text[span[1]:]
    after = _card_span(text, card["after"])
    if after is None:
        raise SystemExit(f"{card['id']}: anchor card {card['after']} not found")
    return text[:after[1]] + "\n\n" + rendered + text[after[1]:]


def place_sidebar(text: str, card: dict) -> str:
    nav_start = text.find('<nav id="sidebar">')
    nav_end = text.find("</nav>", nav_start)
    nav = text[nav_start:nav_end]
    entry = f'<a href="#{card["id"]}">{card["sidebar"]}</a>'
    existing = re.search(rf'\s*<a href="#{re.escape(card["id"])}">[^<]*</a>', nav)
    if existing:
        nav = nav[:existing.start()] + "\n  " + entry + nav[existing.end():]
    else:
        anchor = card.get("sidebar_after", card["after"])
        prev = re.search(rf'<a href="#{re.escape(anchor)}">[^<]*</a>', nav)
        if prev is None:
            group_label = card.get("sidebar_group")
            idx = nav.index(group_label) + len(group_label)
            nav = nav[:idx] + "\n  " + entry + nav[idx:]
        else:
            nav = nav[:prev.end()] + "\n  " + entry + nav[prev.end():]
    return text[:nav_start] + nav + text[nav_end:]


def derive_counts(text: str, version: str) -> str:
    text = re.sub(r"v\d+\.\d+\.\d+", f"v{version}", text)
    total = text.count('<article class="alg-card"')
    text = re.sub(r'<span class="meta-chip">\d+ algorithms</span>',
                  f'<span class="meta-chip">{total} algorithms</span>', text)
    headers = [(m.start(), m.group(1)) for m in re.finditer(r'<span class="group-num">(\d+)</span>', text)]
    counts = {}
    for k, (pos, num) in enumerate(headers):
        end = headers[k + 1][0] if k + 1 < len(headers) else text.find('<section id="methodological-notes"')
        counts[num] = text.count('<article class="alg-card"', pos, end)
    groups = len(headers)
    text = re.sub(r'<span class="meta-chip">\d+ analytical groups</span>',
                  f'<span class="meta-chip">{groups} analytical groups</span>', text)

    # Overview stat tiles: every number is a count of cards, never a typed figure.
    tiles = {
        "Processing Algorithms": total,
        "Analytical Groups": groups,
        "Model &amp; Scenario Tools": counts.get("05", 0),
        "ML &amp; XAI Tools": counts.get("06", 0),
    }
    text = text.replace('<div class="stat-label">Regression Variants</div>',
                        '<div class="stat-label">Model &amp; Scenario Tools</div>')
    for label, value in tiles.items():
        text = re.sub(r'<div class="stat-value">\d+</div><div class="stat-label">' + re.escape(label) + "</div>",
                      f'<div class="stat-value">{value}</div><div class="stat-label">{label}</div>', text)

    def step(match):
        n = counts.get(match.group(1), 0)
        head = match.group(0)[:match.group(0).rindex("<div class=")]
        plural = "s" if n != 1 else ""
        return head + '<div class="step-tools">' + str(n) + " tool" + plural + "</div>"

    text = re.sub(r'<div class="step-num">(\d+)</div>\s*<div class="step-name">[^<]*</div>\s*<div class="step-tools">[^<]*</div>',
                  step, text)
    return text


def build() -> str:
    text = MANUAL.read_text(encoding="utf-8")
    for group in load_groups():
        text = place_group(text, group)
    cards = load_cards()
    titles = sidebar_titles(text)
    titles.update({c["id"]: c["sidebar"] for c in cards})
    for card in cards:
        text = place_card(text, card, titles)
        text = place_sidebar(text, card)
    return derive_counts(text, metadata_version())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    built = build()
    current = MANUAL.read_text(encoding="utf-8")
    if args.check:
        if current != built:
            print("MANUAL BUILD: FAIL (stale: " + MANUAL.name + ") - run py -3 tools/build_manual.py")
            return 1
        print("MANUAL BUILD: OK")
        return 0
    MANUAL.write_text(built, encoding="utf-8", newline="")
    cards = built.count('<article class="alg-card"')
    print(f"wrote {MANUAL} ({cards} cards, v{metadata_version()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
