import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "rvol"

PACKAGE_DEPENDENCIES = {
    "": set(),
    "domain": {"domain"},
    "application": {"application", "domain"},
    "cli": {
        "cli",
        "composition",
        "domain",
        "evaluation",
        "infrastructure",
        "plotting",
        "reporting",
    },
    "composition": {"application", "composition", "domain", "features", "models"},
    "data": {"data"},
    "diagnostics": {"diagnostics", "features", "market"},
    "estimators": {"estimators"},
    "evaluation": {"domain", "evaluation"},
    "features": {"estimators", "features", "market"},
    "market": {"market"},
    "models": {"domain", "models"},
    "infrastructure": {"domain", "infrastructure"},
    "plotting": {
        "diagnostics",
        "domain",
        "estimators",
        "evaluation",
        "plotting",
        "simulation",
    },
    "reporting": {"domain", "evaluation", "reporting"},
    "simulation": {"simulation"},
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


def test_architecture_rules_cover_every_package():
    package_directories = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert set(PACKAGE_DEPENDENCIES).difference({""}) == package_directories


@pytest.mark.parametrize("package", sorted(PACKAGE_DEPENDENCIES))
def test_clean_architecture_dependency_direction(package: str):
    allowed = PACKAGE_DEPENDENCIES[package]
    violations: list[str] = []
    package_root = PACKAGE_ROOT / package
    paths = package_root.glob("*.py") if not package else package_root.rglob("*.py")
    for path in sorted(paths):
        relative_path = path.relative_to(package_root).as_posix()
        forbidden = rvol_dependencies(path).difference(allowed)
        if forbidden:
            violations.append(f"{relative_path}: {', '.join(sorted(forbidden))}")

    package_name = package or "rvol"
    assert not violations, f"{package_name} has outward dependencies: {'; '.join(violations)}"
