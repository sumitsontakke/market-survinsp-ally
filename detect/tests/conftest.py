"""pytest configuration for the detect module tests.

Excludes the _legacy/ subtree from collection. Those tests were imported
verbatim from the pre-monorepo layout during the Phase 3 migration and
their fixture paths still reference the old tree. They will be revived
during Phase 3.5 fixture-migration, but until then they should not
break CI.

Run them explicitly with:
    pytest detect/tests/_legacy
"""
collect_ignore_glob = ["_legacy/*", "_legacy/**/*"]
collect_ignore      = ["_legacy"]