"""Guards the one convention these route files deliberately break.

Every other module in src/ uses `from __future__ import annotations`. The two
route files with an UploadFile parameter must not, and until now the only thing
protecting that was a comment. A formatter, an IDE quick-fix, or a well-meaning
"apply it everywhere" pass would reintroduce it, and the resulting failure is an
import-time crash whose message points at Pydantic response models rather than
at the import that caused it.

Why it breaks (verified by isolated repro):

  - `from __future__ import annotations` stores annotations as strings, so
    `file: UploadFile` becomes the text "UploadFile" rather than the class.
  - Resolving that string back requires the defining module's globals.
  - @limiter.limit() replaces the handler with a wrapper defined inside
    slowapi.extension. functools.wraps copies __name__/__doc__/__annotations__,
    but __globals__ is a read-only attribute of a function object and cannot be
    copied -- so lookups happen in slowapi's namespace, which never imported
    UploadFile.
  - FastAPI's dependant analysis is left holding an unresolved ForwardRef and
    raises at import: "Invalid args for response field! ... ForwardRef('UploadFile')"

Plain str/int params survive this because Pydantic resolves builtins from its
own namespace; UploadFile is what forces resolution of a concrete class. So the
constraint applies only where UploadFile and a rate limiter meet.

The existing route tests would also fail if the import were added, but with
that same misleading message. This test names the actual cause.
"""

import ast
from pathlib import Path

import pytest

_FUTURE_IMPORT = "from __future__ import annotations"

# Route modules combining an UploadFile parameter with @limiter.limit().
_UPLOAD_ROUTE_FILES = [
    "src/api/routes/voice.py",
    "src/api/routes/ingest.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _future_features(source: str) -> set[str]:
    """Names imported from __future__, via the AST rather than a text search.

    A substring check would match the explanatory NOTE comment in these very
    files, which names the import in order to warn against it -- so the guard
    would fail on the documentation it exists to support.
    """
    tree = ast.parse(source)
    return {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        for alias in node.names
    }


@pytest.mark.parametrize("relative_path", _UPLOAD_ROUTE_FILES)
def test_upload_routes_omit_postponed_annotations(relative_path: str) -> None:
    path = _repo_root() / relative_path
    assert path.exists(), f"{relative_path} not found -- update _UPLOAD_ROUTE_FILES"

    assert "annotations" not in _future_features(path.read_text(encoding="utf-8")), (
        f"{relative_path} must not use `{_FUTURE_IMPORT}`.\n"
        "It has an UploadFile parameter on a @limiter.limit()-decorated route. "
        "Postponed evaluation leaves UploadFile as a string that FastAPI tries to "
        "resolve in slowapi's module namespace instead of this one, and the app "
        "then fails to import with a message about Pydantic response fields. "
        "See this test's module docstring for the full mechanism."
    )


@pytest.mark.parametrize("relative_path", _UPLOAD_ROUTE_FILES)
def test_upload_routes_still_have_the_traits_this_guard_assumes(relative_path: str) -> None:
    """Keeps the guard honest.

    If a file stops taking an UploadFile or stops being rate limited, the
    constraint no longer applies and it should be dropped from the list rather
    than left here enforcing a rule for a reason that has gone away.
    """
    source = (_repo_root() / relative_path).read_text(encoding="utf-8")
    assert "UploadFile" in source, (
        f"{relative_path} no longer has an UploadFile parameter; "
        "remove it from _UPLOAD_ROUTE_FILES."
    )
    assert "@limiter.limit" in source, (
        f"{relative_path} is no longer rate limited; remove it from _UPLOAD_ROUTE_FILES."
    )
