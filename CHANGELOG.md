# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.1.3] - 2025-11-12
### Added

- - Added python missing bindings for `states_icrf` and `states_gcrf` for `KeplerianPropagator` and `SGPPropagator`
  - Added additional tests across various module to improve test coverage. [#36](https://github.com/duncaneddy/brahe/pull/36)

### Changed

- - Refactor `frames.rs` file into submodule with subfiles for long-term maintainability. [#14](https://github.com/duncaneddy/brahe/pull/14)
- - Automatically create and merge PRs for changelog updates [#16](https://github.com/duncaneddy/brahe/pull/16)
- - Auto-merge changelog PRs
  - Auto-merge dependabot PRs
  - Expand dependabot to cover python and rust packages [#26](https://github.com/duncaneddy/brahe/pull/26)
- - Bump package version to `v0.1.3` [#34](https://github.com/duncaneddy/brahe/pull/34)
- - Skip unit test suite on auto-generated changelog PRs. [#44](https://github.com/duncaneddy/brahe/pull/44)

### Fixed

- - PR changelogs were not being incorporated into the package changelog due to main-branch protection [#16](https://github.com/duncaneddy/brahe/pull/16)
- - Stop generation changelog PRs for auto-generated changelog PRs [#26](https://github.com/duncaneddy/brahe/pull/26)
- - Fixed issue with release pipeline release note generation [#38](https://github.com/duncaneddy/brahe/pull/38)
- - Fix auto-merge for changelog PRs by using PAT [#40](https://github.com/duncaneddy/brahe/pull/40)
- - Fix auto-merge workflow to accept PAT owner as actor. [#42](https://github.com/duncaneddy/brahe/pull/42)
- - Fix workflow release step to use workflow PAT and declare base branch [#46](https://github.com/duncaneddy/brahe/pull/46)

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

