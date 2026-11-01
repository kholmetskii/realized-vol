import ast
import pathlib
import re

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "rvol"

# Keep separate rules inside grouped packages: grouping must not weaken boundaries.
PACKAGE_DEPENDENCIES = {
    "": set(),
    "models": {"models"},
    "walkforward": {"walkforward", "models"},
    "evaluation": {"evaluation", "walkforward", "models"},
    "data": set(),
    "data.preparation": {"data.preparation"},
    "data.sources": set(),
    # The command line writes where the shared data layout says.
    "data.sources.dukascopy": {"data.sources.dukascopy", "data"},
    "data.market": {"data.market"},
    # Daily data is found where the shared data layout says.
    "data.volatility": {"data.volatility", "data", "data.market", "estimators"},
    "estimators": {"estimators", "estimators.variance"},
    "estimators.variance": set(),
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
    # Array formulas keep their own dependency boundary inside estimators.
    assert set(PACKAGE_DEPENDENCIES) == packages | {"estimators.variance"}
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


# Packages that score, fit and prepare any series must not know what is forecast.
GENERIC_PACKAGES = ("models", "walkforward", "evaluation", "data.preparation")
DOMAIN_WORDS = re.compile(
    r"\b(?:volatil\w*|variances?|realized|realised|rv|\w*_rv|rv_\w*|log_rv\w*|sessions?|fx"
    r"|ticks?|assets?|markets?|dukascopy|trading|eur/?usd|bid|ask|quotes?|prices?|pips?"
    r"|currenc\w*|annuali[sz]\w*|har|ar\(1\))(?=\W|$)",
    re.IGNORECASE,
)


@pytest.mark.parametrize("package", GENERIC_PACKAGES)
def test_generic_packages_do_not_know_the_forecasting_domain(package: str):
    root = PACKAGE_ROOT.joinpath(*package.split("."))
    found = [
        f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{number}: {match.group()}"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        for match in DOMAIN_WORDS.finditer(line)
    ]
    assert not found, "domain vocabulary in a generic package: " + "; ".join(found)


def test_estimators_do_not_know_about_trading_calendars():
    calendar_words = re.compile(r"\b(?:sessions?|trading|new york|fx|close|open)\b", re.IGNORECASE)
    root = PACKAGE_ROOT / "estimators"
    found = [
        f"{path.name}:{number}: {match.group()}"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        for match in calendar_words.finditer(line)
    ]
    assert not found, "calendar knowledge in estimators: " + "; ".join(found)
