# Mirai — Code Rules & Conventions

This file defines coding standards for the Mirai project.
All contributors (human and AI) must follow these rules.

---

## 1. Type Safety — No Duck Typing

**Use concrete types; avoid `hasattr`, `isinstance` guards on known types.**

When you know the type of an object (e.g., it comes from a typed parameter, a
typed collection, or an ABC subclass), access its attributes directly.

```python
# ❌ Bad — unnecessary duck typing
def get_desc(tool: Any) -> str:
    if hasattr(tool, "definition"):
        defn = tool.definition
        if isinstance(defn, dict):
            return defn.get("description", "")
    return ""

# ✅ Good — trust the type system
def get_desc(tool: BaseTool) -> str:
    return tool.definition.get("description", "")
```

**When to use `hasattr` / `isinstance`:**
- Deserializing external data (JSON, user input, config files)
- Working with truly polymorphic interfaces across unrelated types
- Protocol-based dispatch where multiple shapes are expected

**Never use them** when the type is already known through:
- Function signatures and type annotations
- ABC subclasses with abstract methods/properties
- Dataclass fields

## 2. Type Annotations

- All function signatures must have type annotations (parameters + return).
- Use `from __future__ import annotations` for forward references.
- Prefer concrete types over `Any`. Use `Any` only at true boundaries
  (e.g., JSON payloads, third-party APIs without stubs).
- Use `TYPE_CHECKING` blocks for import-only types to avoid circular imports.

```python
# ✅ Good
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mirai.cron import CronScheduler

def start(scheduler: CronScheduler) -> None: ...
```

## 3. Prefer Simple Over Clever

- Write straightforward code. Avoid metaprogramming, monkey-patching,
  dynamic attribute injection, and `__getattr__` overrides unless absolutely
  necessary.
- If a pattern requires a comment explaining "why this isn't as hacky as it
  looks," it's too hacky. Refactor.
- Favor explicit over implicit. Pass dependencies as parameters, not through
  module-level globals or singletons.

## 4. Error Handling

- Catch specific exceptions, never bare `except:`.
- `except Exception` is acceptable only at top-level boundaries
  (e.g., tool execution wrappers, HTTP handlers).
- Always log the exception with context before swallowing it.

```python
# ❌ Bad
try:
    result = await provider.call(prompt)
except:
    result = "error"

# ✅ Good
try:
    result = await provider.call(prompt)
except ProviderError as exc:
    log.error("provider_call_failed", error=str(exc), model=model)
    raise
```

## 5. Naming Conventions

| Item            | Convention                  | Example                   |
|-----------------|-----------------------------|---------------------------|
| Modules         | `snake_case`                | `agent_loop.py`           |
| Classes         | `PascalCase`                | `AgentLoop`, `IMTool`     |
| Functions/Vars  | `snake_case`                | `build_system_prompt`     |
| Constants       | `UPPER_SNAKE`               | `MAX_RETRIES`             |
| Private         | `_leading_underscore`       | `_build_tools_section`    |
| Test files      | `test_<module>.py`          | `test_prompt.py`          |
| Test classes    | `Test<Feature>`             | `TestBuildToolsSection`   |

## 6. Imports

- Group imports: stdlib → third-party → local, separated by blank lines.
- Use absolute imports (`from mirai.agent.tools.base import BaseTool`),
  not relative (`from .base import BaseTool`).
- One import per line for `from X import` when importing multiple names.

## 7. Logging

- Use structured logging (`structlog` via `mirai.logging.get_logger`).
- Log keys are `snake_case`, no spaces.
- Every log event must have a descriptive event name as first arg.

```python
log.info("cron_job_fired", job_id=job_id, run_count=count)
```

## 8. Async Discipline

- Never call `asyncio.run()` inside async code.
- Never `await` a synchronous function (it will `TypeError`).
- Use `asyncio.create_task()` for fire-and-forget; track the task
  to prevent GC and catch exceptions.

## 9. Tests

- Every new module needs a corresponding `test_<module>.py`.
- Use `pytest` + `pytest-asyncio` (`@pytest.mark.asyncio`).
- Use `MagicMock` / `AsyncMock` for mocking; avoid patching internals
  when constructor injection is available.
- Test names should read as specifications:
  `test_empty_tools_returns_empty_string`, not `test_1`.

## 10. Documentation

- All code and documentation must be written in **English**.
- Every public class and function needs a docstring.
- Use PEP 257 style (one-line summary, optional extended description).
- Keep comments for *why*, not *what* — the code should explain the *what*.
