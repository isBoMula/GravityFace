from pathlib import Path
import re


def write_eval_section(result_dir, section_id, title, body):
    eval_dir = Path(result_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "eval.md"

    section = _format_section(section_id, title, body)
    existing = eval_path.read_text(encoding="utf-8") if eval_path.exists() else ""
    eval_path.write_text(_replace_or_append(existing, section_id, section), encoding="utf-8")
    return str(eval_path)


def _format_section(section_id, title, body):
    body_text = str(body).rstrip()
    return (
        f"<!-- BEGIN EVAL: {section_id} -->\n"
        f"## {title}\n\n"
        "```text\n"
        f"{body_text}\n"
        "```\n"
        f"<!-- END EVAL: {section_id} -->\n"
    )


def _replace_or_append(existing, section_id, section):
    start = f"<!-- BEGIN EVAL: {section_id} -->"
    end = f"<!-- END EVAL: {section_id} -->"
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}\n?", re.DOTALL)

    if pattern.search(existing):
        return pattern.sub(section, existing)

    if existing.strip():
        return existing.rstrip() + "\n\n" + section

    return section
