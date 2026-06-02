# Releasing Col-Bandit to PyPI

## One-time setup
1. **Rotate any previously shared tokens.** Treat any token that has ever been pasted into chat / email / Slack as compromised.
2. **Create a fresh PyPI API token** at https://pypi.org/manage/account/token/ — initial scope "Entire account" (narrow to the `colbandit` / `numkong` projects after the first successful publish).
3. **Add it as a GitHub Actions secret** in `roipony/ColBandit`:
   - Repo → Settings → Secrets and variables → Actions → "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: the raw `pypi-…` token

That's the only place the token should live.

## Cutting a release
```bash
# 1) bump version
sed -i 's/^__version__ = .*/__version__ = "0.1.0"/' colbandit/_version.py
# match the version on numkong (native/numkong/pyproject.toml) if you ship a numkong release too

# 2) tag + push
git tag v0.1.0
git push origin v0.1.0
```
The `.github/workflows/build-wheels.yml` workflow then:
1. Builds **numkong** wheels for **Linux x86_64, macOS arm64, macOS x86_64, Windows** (Python 3.9–3.12) using `cibuildwheel`.
2. Builds the **colbandit** pure-Python wheel + sdist.
3. Uploads **all** artifacts to PyPI (`pypa/gh-action-pypi-publish`).

End-users then get one-command install on any of those platforms:
```bash
pip install colbandit
```
Pip auto-selects the correct precompiled numkong wheel — no Rust toolchain on the user's machine.

## Dry runs (recommended before going live)
Uncomment the `repository-url: https://test.pypi.org/legacy/` line in the publish step and use a TestPyPI token. Then:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ colbandit
```

## Notes
- **Anonymity / ARR:** the paper is mid-review and Appendix D promises "release upon publication." Until acceptance, prefer TestPyPI; flip to production PyPI after the decision.
- **Name collisions:** if `colbandit` or `numkong` is taken on PyPI, add a vendor prefix (`ibm-colbandit`, `ibm-numkong`) — change `name = "…"` in each `pyproject.toml` (and the `dependencies = ["…"]` line in colbandit/pyproject.toml).
- **Apple Silicon vs Intel Mac:** `macos-14` runner produces `*_arm64.whl`; `macos-13` produces `*_x86_64.whl`. End users on either Mac get the right wheel automatically.
