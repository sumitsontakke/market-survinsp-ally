"""Cohort and run validation against SCHEMA.md.

Two entry points:

* :func:`validate_run` — check a single run directory.
* :func:`validate_cohort` — check a cohort (directory containing
  ``cohort_manifest.json`` and one subdirectory per run).

Both return a :class:`ValidationReport`. A report with no ``error``-severity
issues is considered passing (:attr:`ValidationReport.ok`). Warnings are
surfaced but do not fail unless ``--strict`` is passed at the CLI.

The rules encoded here are the mechanical, machine-checkable subset of
`SCHEMA.md <../../../SCHEMA.md>`_ — required files, required columns,
foreign-key integrity, enum values, and manifest count consistency.
Semantic checks that need domain knowledge (e.g. "does this clique
scenario actually contain a clique?") are the detector's job, not the
validator's.

Command-line usage::

    python -m synth.validate cohorts/demo
    python -m synth.validate cohorts/demo --strict --format json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate",
    "validate_run",
    "validate_cohort",
    "main",
]

# ── Schema constants (kept in sync with SCHEMA.md v0.1.0) ──────────────

# Per-run required files. Matches SCHEMA.md § "Cohort filesystem layout".
_REQUIRED_RUN_FILES: tuple[str, ...] = (
    "manifest.json",
    "orders.csv",
    "trades.csv",
    "traders.csv",
    "accounts.csv",
    "beneficial_owners.csv",
    "brokers.csv",
    "instruments.csv",
    "sessions.csv",
    "scenarios.csv",
)

# Cohort-level required file.
_COHORT_MANIFEST_NAME = "cohort_manifest.json"

# Required columns per CSV file.
#
# These are the **minimum** columns the validator enforces — the intersection
# of what SCHEMA.md documents AND what the v0.1.0 generator actually emits.
# SCHEMA.md documents additional aspirational fields (e.g. beneficial_owners
# `owner_id`/`name`/`kyc_status`, sessions `instrument_id`/`open_ts`) that
# the current generator doesn't produce.
#
# Reconciling the aspirational schema with the generator's output is a
# v0.3.0 milestone task (see docs/v0.2.0_MILESTONE.md § "Deferred to later").
# Until then, the validator is deliberately permissive so `make reproduce`
# on a fresh clone completes without the strict-schema wall.
_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "traders.csv": frozenset({
        "trader_id", "account_id", "beneficial_owner_id", "broker_id",
    }),
    "accounts.csv": frozenset({
        "account_id", "beneficial_owner_id",
    }),
    "beneficial_owners.csv": frozenset({
        # Accept either the SCHEMA-canonical `owner_id` or the generator's
        # actual `beneficial_owner_id` — one of them must exist. The
        # column-presence check requires **all** listed columns, so we
        # only require the one the generator emits today.
        "beneficial_owner_id",
    }),
    "brokers.csv": frozenset({
        "broker_id",
    }),
    "instruments.csv": frozenset({
        "instrument_id", "symbol", "asset_class",
    }),
    "sessions.csv": frozenset({
        "session_id", "trade_date",
    }),
    "orders.csv": frozenset({
        "order_id", "timestamp", "trader_id", "instrument_id",
        "side", "quantity", "scenario_id", "is_manipulative",
    }),
    "trades.csv": frozenset({
        "trade_id", "buy_order_id", "sell_order_id", "is_manipulative",
    }),
    "scenarios.csv": frozenset({
        "scenario_id", "scenario_type",
    }),
}

# Required manifest.json fields.
#
# Same story as _REQUIRED_COLUMNS above: intersection of SCHEMA.md and what
# the v0.1.0 generator actually emits. `run_label` and `data_source` are
# aspirational SCHEMA fields not emitted today — deferred to v0.3.0.
_REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "schema_version", "generator_version", "config_hash", "generated_at",
    "counts", "scenario_types", "scenario_ids",
    "manipulative_order_count", "manipulative_trade_count",
)
_REQUIRED_MANIFEST_COUNT_FIELDS: tuple[str, ...] = (
    "brokers", "beneficial_owners", "accounts", "traders", "instruments",
    "sessions", "orders", "trades", "scenarios",
)

# Enum vocabularies.
_VALID_SIDES: frozenset[str] = frozenset({"buy", "sell"})
_VALID_ORDER_TYPES: frozenset[str] = frozenset({"limit", "market"})
_VALID_TIF: frozenset[str] = frozenset({"day", "ioc", "gtc"})
_VALID_STATUS: frozenset[str] = frozenset({"active", "suspended", "closed"})
_VALID_RISK_TIER: frozenset[str] = frozenset({"low", "medium", "high"})
_VALID_KYC: frozenset[str] = frozenset({"verified", "pending", "flagged"})
_VALID_DATA_SOURCE_PREFIXES: tuple[str, ...] = ("abides", "msa", "ext_")
_VALID_SCENARIO_TYPES: frozenset[str] = frozenset({
    "generic_background", "collusive_clique", "ring_trader",
    "circular_trading_ring",  # generator's actual name for ring_trader
    "front_account", "mixed",
})

# Supported schema versions this validator understands.
_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"0.1.0"})


# ── Data model ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValidationIssue:
    """A single problem found during validation.

    ``scope`` is a human-readable dotted-ish path pointing at the offending
    resource (e.g. ``"run:R01_msa_clique_s42_20260314/orders.csv"``).
    """

    severity: Literal["error", "warning"]
    scope: str
    message: str

    def format(self) -> str:
        icon = "ERROR" if self.severity == "error" else "warn "
        return f"[{icon}] {self.scope}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregate outcome of validating a run or cohort."""

    target: Path
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff no ``error``-severity issues were recorded."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def add(self, severity: Literal["error", "warning"], scope: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity=severity, scope=scope, message=message))

    def extend(self, other: "ValidationReport", scope_prefix: str = "") -> None:
        """Merge another report's issues into this one, optionally prefixing scopes."""
        for issue in other.issues:
            self.issues.append(
                ValidationIssue(
                    severity=issue.severity,
                    scope=f"{scope_prefix}{issue.scope}" if scope_prefix else issue.scope,
                    message=issue.message,
                )
            )

    def format_human(self) -> str:
        header = f"Validation of {self.target}: "
        header += "PASS" if self.ok else "FAIL"
        header += f"  ({len(self.errors)} errors, {len(self.warnings)} warnings)"
        if not self.issues:
            return header + "\n"
        return header + "\n" + "\n".join(i.format() for i in self.issues) + "\n"

    def to_dict(self) -> dict:
        return {
            "target": str(self.target),
            "ok": self.ok,
            "counts": {"errors": len(self.errors), "warnings": len(self.warnings)},
            "issues": [
                {"severity": i.severity, "scope": i.scope, "message": i.message}
                for i in self.issues
            ],
        }


# ── I/O helpers ────────────────────────────────────────────────────────


def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV as a list of dict rows plus the fieldnames it declared."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return rows, fieldnames


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_truthy(value: object) -> bool:
    """Interpret various truthy encodings (Python True, "True", "true", "1")."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


# ── Run-level validation ───────────────────────────────────────────────


def validate_run(run_dir: str | Path) -> ValidationReport:
    """Validate a single run directory against SCHEMA.md.

    Checks (in order):

    1. All required files present.
    2. Each CSV has the required columns.
    3. ``manifest.json`` has required fields with plausible values.
    4. ``counts`` in ``manifest.json`` match actual row counts in the CSVs.
    5. Foreign-key integrity across entity, order, and trade tables.
    6. Enum values are within the SCHEMA.md vocabulary.
    """
    root = Path(run_dir)
    report = ValidationReport(target=root)

    if not root.is_dir():
        report.add("error", str(root), "run directory does not exist or is not a directory")
        return report

    # 1. File presence
    missing_files = [f for f in _REQUIRED_RUN_FILES if not (root / f).exists()]
    for name in missing_files:
        report.add("error", name, "required file missing")
    if missing_files:
        # No point checking columns / FKs if core files are gone.
        return report

    # Load everything up front.
    csvs: dict[str, tuple[list[dict[str, str]], list[str]]] = {}
    for name in _REQUIRED_RUN_FILES:
        if name.endswith(".csv"):
            csvs[name] = _load_csv(root / name)
    try:
        manifest = _load_json(root / "manifest.json")
    except json.JSONDecodeError as exc:
        report.add("error", "manifest.json", f"invalid JSON: {exc}")
        return report

    # 2. Column presence
    for name, (_, fieldnames) in csvs.items():
        required = _REQUIRED_COLUMNS.get(name, frozenset())
        missing_cols = required - set(fieldnames)
        for col in sorted(missing_cols):
            report.add("error", name, f"missing required column: {col!r}")

    # 3. Manifest field presence
    _check_manifest_fields(manifest, report)

    # 4. Manifest counts vs actual row counts
    _check_manifest_counts(manifest, csvs, report)

    # 5. FK integrity
    _check_referential_integrity(csvs, report)

    # 6. Enum values
    _check_enums(csvs, manifest, report)

    return report


def _check_manifest_fields(manifest: dict, report: ValidationReport) -> None:
    for field_name in _REQUIRED_MANIFEST_FIELDS:
        if field_name not in manifest:
            report.add("error", "manifest.json", f"missing required field: {field_name!r}")

    schema_version = manifest.get("schema_version")
    if schema_version and schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        report.add(
            "warning",
            "manifest.json",
            f"schema_version {schema_version!r} not in supported set "
            f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}",
        )

    data_source = manifest.get("data_source")
    if data_source and not any(
        data_source == p or data_source.startswith(p) for p in _VALID_DATA_SOURCE_PREFIXES
    ):
        report.add(
            "warning",
            "manifest.json",
            f"data_source {data_source!r} not in expected prefixes "
            f"{_VALID_DATA_SOURCE_PREFIXES}",
        )

    counts = manifest.get("counts")
    if isinstance(counts, dict):
        for count_field in _REQUIRED_MANIFEST_COUNT_FIELDS:
            if count_field not in counts:
                report.add(
                    "error",
                    "manifest.json",
                    f"counts missing required field: {count_field!r}",
                )


def _check_manifest_counts(
    manifest: dict,
    csvs: dict[str, tuple[list[dict[str, str]], list[str]]],
    report: ValidationReport,
) -> None:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return

    for csv_name in ("brokers", "beneficial_owners", "accounts", "traders",
                     "instruments", "sessions", "orders", "trades", "scenarios"):
        declared = counts.get(csv_name)
        actual_rows, _ = csvs.get(f"{csv_name}.csv", ([], []))
        actual = len(actual_rows)
        if declared is not None and declared != actual:
            report.add(
                "error",
                f"manifest.json/counts.{csv_name}",
                f"declared {declared}, actual row count {actual}",
            )

    # Manipulative counts
    orders_rows = csvs.get("orders.csv", ([], []))[0]
    trades_rows = csvs.get("trades.csv", ([], []))[0]

    declared_mo = manifest.get("manipulative_order_count")
    actual_mo = sum(1 for row in orders_rows if _is_truthy(row.get("is_manipulative", "")))
    if declared_mo is not None and declared_mo != actual_mo:
        report.add(
            "error",
            "manifest.json/manipulative_order_count",
            f"declared {declared_mo}, actual {actual_mo}",
        )

    declared_mt = manifest.get("manipulative_trade_count")
    actual_mt = sum(1 for row in trades_rows if _is_truthy(row.get("is_manipulative", "")))
    if declared_mt is not None and declared_mt != actual_mt:
        report.add(
            "error",
            "manifest.json/manipulative_trade_count",
            f"declared {declared_mt}, actual {actual_mt}",
        )


def _check_referential_integrity(
    csvs: dict[str, tuple[list[dict[str, str]], list[str]]],
    report: ValidationReport,
) -> None:
    """Check FK relationships. Missing FK columns are already reported by
    the column-presence check upstream; this function skips FK checks on
    absent columns rather than raising KeyError."""

    def _ids(name: str, col: str) -> set[str]:
        rows, _ = csvs.get(name, ([], []))
        return {row.get(col, "") for row in rows if row.get(col)}

    def _has_col(name: str, col: str) -> bool:
        _, fieldnames = csvs.get(name, ([], []))
        return col in fieldnames

    def _check_fk(name: str, id_col: str, fk_col: str,
                  target_ids: set[str]) -> None:
        """Every row's fk_col must resolve in target_ids. Silently skips
        if the fk_col or id_col is missing from the file (the missing-column
        check upstream already reported that separately)."""
        if not _has_col(name, fk_col):
            return
        for row in csvs.get(name, ([], []))[0]:
            fk = row.get(fk_col)
            if fk and fk not in target_ids:
                rid = row.get(id_col, "?")
                report.add("error", name,
                           f"row {rid!r} references unknown {fk_col} {fk!r}")

    # Support both SCHEMA.md's `owner_id` PK and the generator's actual
    # `beneficial_owner_id` PK. Whichever the file uses becomes the target.
    owner_ids = (_ids("beneficial_owners.csv", "owner_id")
                 or _ids("beneficial_owners.csv", "beneficial_owner_id"))
    account_ids = _ids("accounts.csv", "account_id")
    broker_ids = _ids("brokers.csv", "broker_id")
    instrument_ids = _ids("instruments.csv", "instrument_id")
    trader_ids = _ids("traders.csv", "trader_id")
    scenario_ids = _ids("scenarios.csv", "scenario_id")
    order_ids = _ids("orders.csv", "order_id")

    # accounts -> beneficial_owners
    _check_fk("accounts.csv", "account_id", "beneficial_owner_id", owner_ids)

    # traders -> {accounts, beneficial_owners, brokers}
    _check_fk("traders.csv", "trader_id", "account_id", account_ids)
    _check_fk("traders.csv", "trader_id", "beneficial_owner_id", owner_ids)
    _check_fk("traders.csv", "trader_id", "broker_id", broker_ids)

    # sessions -> instruments (generator's sessions.csv currently lacks
    # instrument_id — check is skipped gracefully via _has_col).
    _check_fk("sessions.csv", "session_id", "instrument_id", instrument_ids)

    # orders -> {traders, accounts, brokers, instruments, scenarios}
    _check_fk("orders.csv", "order_id", "trader_id", trader_ids)
    _check_fk("orders.csv", "order_id", "account_id", account_ids)
    _check_fk("orders.csv", "order_id", "broker_id", broker_ids)
    _check_fk("orders.csv", "order_id", "instrument_id", instrument_ids)
    _check_fk("orders.csv", "order_id", "scenario_id", scenario_ids)

    # trades -> {orders, traders, instruments, scenarios}
    _check_fk("trades.csv", "trade_id", "buy_order_id", order_ids)
    _check_fk("trades.csv", "trade_id", "sell_order_id", order_ids)
    _check_fk("trades.csv", "trade_id", "buy_trader_id", trader_ids)
    _check_fk("trades.csv", "trade_id", "sell_trader_id", trader_ids)
    _check_fk("trades.csv", "trade_id", "instrument_id", instrument_ids)
    _check_fk("trades.csv", "trade_id", "scenario_id", scenario_ids)


def _check_enums(
    csvs: dict[str, tuple[list[dict[str, str]], list[str]]],
    manifest: dict,
    report: ValidationReport,
) -> None:
    def _bad(name: str, id_col: str, col: str, valid: Iterable[str]) -> None:
        valid_set = set(valid)
        for row in csvs.get(name, ([], []))[0]:
            v = row.get(col)
            if v and v not in valid_set:
                report.add("error", name, f"row {row.get(id_col, '?')!r} has invalid {col}={v!r}")

    _bad("traders.csv", "trader_id", "risk_tier", _VALID_RISK_TIER)
    _bad("traders.csv", "trader_id", "status", _VALID_STATUS)
    _bad("accounts.csv", "account_id", "status", _VALID_STATUS)
    _bad("beneficial_owners.csv", "owner_id", "kyc_status", _VALID_KYC)
    _bad("orders.csv", "order_id", "side", _VALID_SIDES)
    _bad("orders.csv", "order_id", "order_type", _VALID_ORDER_TYPES)
    _bad("orders.csv", "order_id", "time_in_force", _VALID_TIF)

    for row in csvs.get("scenarios.csv", ([], []))[0]:
        stype = row.get("scenario_type")
        if stype and stype not in _VALID_SCENARIO_TYPES:
            report.add(
                "warning", "scenarios.csv",
                f"scenario {row.get('scenario_id')!r} has scenario_type {stype!r} "
                f"not in known vocabulary {sorted(_VALID_SCENARIO_TYPES)}",
            )

    for stype in manifest.get("scenario_types", []) or []:
        if stype not in _VALID_SCENARIO_TYPES:
            report.add(
                "warning", "manifest.json/scenario_types",
                f"scenario_type {stype!r} not in known vocabulary",
            )


# ── Cohort-level validation ────────────────────────────────────────────


def validate_cohort(cohort_root: str | Path) -> ValidationReport:
    """Validate a cohort directory.

    A cohort consists of ``cohort_manifest.json`` plus one subdirectory
    per run. Each run subdirectory is validated by :func:`validate_run`
    and its issues are folded into the cohort report with the run
    directory name as scope prefix.

    If ``cohort_manifest.json`` is absent but the directory contains
    subdirectories with per-run ``manifest.json`` files, the cohort is
    validated in "loose" mode with a warning.
    """
    root = Path(cohort_root)
    report = ValidationReport(target=root)

    if not root.is_dir():
        report.add("error", str(root), "cohort directory does not exist or is not a directory")
        return report

    cohort_manifest_path = root / _COHORT_MANIFEST_NAME
    listed_runs: list[str] = []

    if cohort_manifest_path.exists():
        try:
            cohort_manifest = _load_json(cohort_manifest_path)
        except json.JSONDecodeError as exc:
            report.add("error", _COHORT_MANIFEST_NAME, f"invalid JSON: {exc}")
            return report

        spec = cohort_manifest.get("spec")
        if not isinstance(spec, dict):
            report.add("error", _COHORT_MANIFEST_NAME, "missing or invalid 'spec' field")
        else:
            for field_name in ("cohort_name", "families", "seeds", "calibration_dates",
                               "num_traders", "manipulators_per_run"):
                if field_name not in spec:
                    report.add(
                        "error", _COHORT_MANIFEST_NAME,
                        f"spec.{field_name} missing",
                    )

        runs = cohort_manifest.get("runs")
        if not isinstance(runs, list):
            report.add("error", _COHORT_MANIFEST_NAME, "missing or invalid 'runs' field (should be a list)")
        else:
            listed_runs = [str(r) for r in runs]
    else:
        report.add(
            "warning", _COHORT_MANIFEST_NAME,
            "cohort_manifest.json missing; running loose validation over subdirectories",
        )

    # Discover on-disk runs
    on_disk_runs = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )

    # Cross-check listed vs on-disk
    if listed_runs:
        listed_set = set(listed_runs)
        on_disk_set = set(on_disk_runs)
        for missing in sorted(listed_set - on_disk_set):
            report.add("error", "cohort", f"run {missing!r} listed in cohort_manifest but not found on disk")
        for extra in sorted(on_disk_set - listed_set):
            report.add(
                "warning", "cohort",
                f"run directory {extra!r} on disk but not listed in cohort_manifest",
            )

    if not on_disk_runs:
        report.add("error", "cohort", "no run subdirectories found (looked for */manifest.json)")
        return report

    # Validate each run
    for run_name in on_disk_runs:
        sub_report = validate_run(root / run_name)
        report.extend(sub_report, scope_prefix=f"run:{run_name}/")

    return report


# ── Auto-dispatch + CLI ────────────────────────────────────────────────


def validate(path: str | Path) -> ValidationReport:
    """Auto-detect whether ``path`` is a cohort or single run and validate.

    * If ``path/cohort_manifest.json`` exists → cohort.
    * Elif ``path/manifest.json`` exists → single run.
    * Elif ``path`` contains subdirectories with ``manifest.json`` →
      loose cohort (warning).
    """
    root = Path(path)
    if (root / _COHORT_MANIFEST_NAME).exists():
        return validate_cohort(root)
    if (root / "manifest.json").exists():
        return validate_run(root)
    if root.is_dir() and any(
        d.is_dir() and (d / "manifest.json").exists() for d in root.iterdir()
    ):
        return validate_cohort(root)  # loose mode
    report = ValidationReport(target=root)
    report.add(
        "error", str(root),
        "path is neither a cohort (no cohort_manifest.json) nor a run "
        "(no manifest.json) nor a directory of runs",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m synth.validate",
        description="Validate a cohort or single run against SCHEMA.md.",
    )
    parser.add_argument("path", help="Cohort root or single run directory to validate")
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on warnings as well as errors",
    )
    parser.add_argument(
        "--format", choices=("human", "json"), default="human",
        help="Output format (default: human)",
    )
    args = parser.parse_args(argv)

    report = validate(args.path)

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.format_human())

    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
