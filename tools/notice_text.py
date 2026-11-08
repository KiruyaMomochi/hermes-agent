"""Notice templates shared by producers and transcript classifiers."""

from tools.prompt_text import tool_text

NOTICE_TEMPLATES = {
    "async_delegation.header": "[ASYNC DELEGATION COMPLETE — {deleg_id}]",
    "async_delegation.batch_header": "[ASYNC DELEGATION BATCH COMPLETE — {deleg_id}]",
    "async_delegation.task_failed_header": "[ASYNC DELEGATION TASK FAILED — {deleg_id}, task {task_number}/{n_tasks}]",
    "process.watch_header": '[IMPORTANT: Background process {session_id} matched watch pattern "{pattern}".\n',
    "process.completion": (
        "[IMPORTANT: Background process {session_id} {status} (exit code {exit_code}{signal}).\n"
        "{attribution}Command: {command}\nOutput:\n{output}]"),
}


def notice_text(kind: str) -> str:
    """Use the existing profile cache and delegated-child default policy."""
    return tool_text(f"notifications.{kind}", NOTICE_TEMPLATES[kind])


def notice_prefixes(family: str | None = None) -> tuple[str, ...]:
    """Current and original literal prefixes; never let an empty prefix match all input.

    Result wording is live, unlike pinned schemas. tool_text reuses the YAML cache
    keyed by profile path/mtime/size, so transcript scans never parse it per row.
    """
    return tuple(dict.fromkeys(
        prefix for kind, default in NOTICE_TEMPLATES.items()
        if family is None or kind.startswith(family + ".")
        for template in (default, notice_text(kind))
        if (prefix := template.partition("{")[0].lstrip())
    ))
