# Homebrew and Direct Binary Distribution (`te-rs`)

This document describes how to distribute `te-rs` binaries produced by CI release artifacts.

## Release Artifact Inputs

`rust-artifacts.yml` publishes archives named:

- `te-rs-<OS>-<ARCH>.tar.gz`

Example expected assets for a GitHub release tag `vX.Y.Z`:

- `te-rs-Linux-X64.tar.gz`
- `te-rs-macOS-ARM64.tar.gz`

Each asset should have a published SHA256 checksum.

## Direct Install (No Homebrew)

```bash
VERSION=vX.Y.Z
ASSET=te-rs-macOS-ARM64.tar.gz
curl -L -o "$ASSET" "https://github.com/<org>/<repo>/releases/download/$VERSION/$ASSET"
shasum -a 256 "$ASSET"
tar -xzf "$ASSET"
install -m 0755 te-rs-macOS-ARM64 /usr/local/bin/te-rs
te-rs --help
```

## Homebrew Formula Workflow

If maintaining a dedicated tap (recommended):

1. Create/update `Formula/te-rs.rb` in the tap repository.
2. Point `url` to the release asset for the target platform.
3. Set `sha256` from published checksums.
4. Bump `version` to match release tag.
5. Merge and validate with:

```bash
brew update
brew install <tap>/te-rs
te-rs --help
```

## Smoke Test Checklist

Run on macOS after publishing artifacts:

```bash
te-rs --help
te-rs health
te-rs index files . --query "deprecated" --top-k 1
```

## Ownership

If the tap is separate from the main repository, record:

- tap repo URL
- maintainers responsible for formula updates
- release-to-formula update SLA
