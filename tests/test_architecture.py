import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "rvol"

LAYER_DEPENDENCIES = {
    "domain": {"domain"},
    "application": {"application", "domain"},
    "evaluation": {"domain", "evaluation"},
    "models": {"domain", "models"},
    "infrastructure": {"domain", "infrastructure"},
    "reporting": {"domain", "evaluation", "reporting"},
}

# This module preserves the pre-refactor public API by delegating to the new
# application service and model adapters. No new evaluation module gets this exception.
FILE_DEPENDENCY_EXCEPTIONS = {
    ("evaluation", "walk_forward.py"): {"application", "models"},
}


def rvol_dependencies(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        for module in modules:
            if module == "rvol":
                dependencies.add("")
            elif module.startswith("rvol."):
                dependencies.add(module.split(".")[1])
    return dependencies


@pytest.mark.parametrize("layer", sorted(LAYER_DEPENDENCIES))
def test_clean_architecture_dependency_direction(layer):
    allowed = LAYER_DEPENDENCIES[layer]
    violations: list[str] = []
    for path in sorted((PACKAGE_ROOT / layer).glob("*.py")):
        file_allowed = allowed | FILE_DEPENDENCY_EXCEPTIONS.get((layer, path.name), set())
        forbidden = rvol_dependencies(path).difference(file_allowed)
        if forbidden:
            violations.append(f"{path.name}: {', '.join(sorted(forbidden))}")

    assert not violations, f"{layer} has outward dependencies: {'; '.join(violations)}"
