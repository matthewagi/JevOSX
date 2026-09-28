"""Human-readable text map of an Observation (what `jevosx observe` prints)."""

from __future__ import annotations

from ..types import Observation, clean_text


def render_text_map(obs: Observation, *, menus: bool = False, text_lines: int = 12) -> str:
    app = obs.app
    lines = [f"▣ {app.name} ({app.bundle_id or 'no bundle id'}, pid {app.pid})"]
    if obs.window is not None:
        others = [w.title for w in obs.windows if not w.focused]
        suffix = f"  · other windows: {', '.join(others[:5])}" if others else ""
        lines.append(f'  window "{obs.window.title}"{suffix}')
    stats = obs.stats
    lines.append(
        f"  {len(obs.elements)} elements · {len(obs.scroll_areas)} scroll areas · {len(obs.menu_items)} menu commands"
        f" · visited {stats.get('visited', '?')} nodes in {stats.get('walk_ms', '?')} ms"
        + (" · TRUNCATED" if stats.get("truncated") else "")
    )
    vision = stats.get("vision")
    if vision:
        if vision.get("ran"):
            lines.append(
                f"  vision: {vision.get('elements', 0)} on-screen texts read by OCR in {vision.get('ms', '?')} ms"
                + (" (unchanged image, reused)" if vision.get("cached") else "")
            )
        else:
            lines.append(f"  vision: not used ({vision.get('note', 'unavailable')})")
    lines.append("")
    current = object()
    for element in obs.elements:
        if element.container != current:
            current = element.container
            lines.append(f"  ┌ {element.container or 'window'}")
        flags = element.states()
        ops = ",".join(element.ops) if element.ops else "-"
        lines.append(f"  │ {element.describe()}" + (f"  ⟨{', '.join(flags)}⟩" if flags else "") + f"  {{{ops}}}")
    if obs.scroll_areas:
        lines.append("")
        for area in obs.scroll_areas:
            lines.append(f"  [s{area.index}] scroll area {area.label}")
    if menus and obs.menu_items:
        lines.append("")
        for item in obs.menu_items:
            lines.append(f"  [m{item.index}] {item.label}" + (f"  ({item.shortcut})" if item.shortcut else ""))
    elif obs.menu_items:
        lines.append(f"\n  ({len(obs.menu_items)} menu commands hidden; pass --menus to list them)")
    if obs.text:
        lines.append("")
        lines.append("  visible text:")
        for line in obs.text.splitlines()[:text_lines]:
            lines.append(f"    {clean_text(line, 110)}")
    return "\n".join(lines)
