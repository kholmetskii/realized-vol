import ast
import pathlib

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "rvol"

# Keep separate rules inside grouped packages: grouping must not weaken boundaries.
PACKAGE_DEPENDENCIES = {
    "": set(),
    "models": {"models"},
    "backtesting": {"backtesting", "models"},
    "evaluation": {"evaluation", "backtesting", "models"},
    "evaluation.volatility": {"evaluation", "backtesting", "models"},
    "data": set(),
    "data.preparation": set(),
    "data.sources": set(),
    "data.market": {"data.market"},
    "data.volatility": {"data.volatility", "data.market", "estimators"},
    "estimators": {"estimators", "estimators.variance", "data.market"},
    "estimators.variance": set(),
    "simulation": {"simulation"},
    "experiment": {
        "experiment",
        "backtesting",
        "data.preparation",
        "data.volatility",
        "evaluation",
        "evaluation.volatility",
        "models",
    },
}


def package_boundary(module: str) -> str:
    matches = [
        name
        for name in PACKAGE_DEPENDENCIES
        if name and (module == name or module.startswith(name + "."))
    ]
    return max(matches, key=len, default="")


def source_module(path: pathlib.Path) -> str:
    parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


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
                dependencies.add(module.removeprefix("rvol."))
    return dependencies


def test_architecture_rules_cover_every_package():
    packages = {source_module(path) for path in PACKAGE_ROOT.rglob("__init__.py")}
    # Optional preparation and domain metrics keep their own dependency boundaries.
    assert set(PACKAGE_DEPENDENCIES) == packages | {
        "data.preparation",
        "estimators.variance",
        "evaluation.volatility",
        "experiment",
    }
    for name in PACKAGE_DEPENDENCIES:
        path = PACKAGE_ROOT.joinpath(*name.split(".")) if name else PACKAGE_ROOT
        assert (path / "__init__.py").is_file() or path.with_suffix(".py").is_file()


@pytest.mark.parametrize("package", sorted(PACKAGE_DEPENDENCIES))
def test_clean_architecture_dependency_direction(package: str):
    allowed = PACKAGE_DEPENDENCIES[package]
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if package_boundary(source_module(path)) != package:
            continue
        forbidden = {
            dependency
            for dependency in rvol_dependencies(path)
            if package_boundary(dependency) not in allowed
        }
        if forbidden:
            relative_path = path.relative_to(PACKAGE_ROOT).as_posix()
            violations.append(f"{relative_path}: {', '.join(sorted(forbidden))}")
    package_name = package or "rvol"
    assert not violations, f"{package_name} has outward dependencies: {'; '.join(violations)}"


def test_models_have_no_required_specification_or_domain_layer():
    assert not (PACKAGE_ROOT / "domain").exists()
    for package in ("models", "backtesting"):
        for path in (PACKAGE_ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            assert not any(
                isinstance(node, ast.FunctionDef) and node.name == "specification"
                for node in ast.walk(tree)
            ), str(path)
