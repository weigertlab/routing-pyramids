# Contributing guide

```sh
# format
uv run ruff format
# lint
uv run ruff check
# type check
uv run pyrefly check
# test
uv run pytest
```

## Releasing

Publishing uses PyPI trusted publishing with GitHub Actions.
For each release, update and commit the version and lockfile:

```sh
uv version 0.2.0
git add pyproject.toml uv.lock
git commit -m "release 0.2.0"
```

Wait for CI to pass on the version commit,
then create and push the matching version tag.
The tag must be the package version prefixed with `v`.
Pushing it starts the publishing [workflow](.github/workflows/publish.yml).
