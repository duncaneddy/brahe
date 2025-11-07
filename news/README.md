# Changelog Fragments

This directory contains changelog fragments that will be compiled into `CHANGELOG.md` during the release process.

## Automatic Fragment Creation

When a pull request is merged, changelog fragments are **automatically created** from the PR description. You don't need to manually create fragment files in most cases.

The GitHub Actions workflow will:
1. Parse the changelog section in the PR description
2. Create fragment files in this directory (e.g., `123.added.md`, `123.fixed.md`)
3. Commit the fragments to the main branch

## Manual Fragment Creation (Backup Method)

If you need to manually create a fragment, use this format:

### Filename Convention
```
<PR#>.<type>.md
```

Where `<type>` is one of:
- `added` - New features
- `changed` - Changes in existing functionality
- `fixed` - Bug fixes
- `removed` - Removed features

### Examples
```
123.added.md
124.fixed.md
125.changed.md
```

### Fragment Content

Each fragment file should contain one or more bullet points describing the changes:

```markdown
- Added support for new orbital propagator
- Added EOP data caching functionality
```

Or for a single change:

```markdown
- Fixed memory leak in trajectory interpolation
```

## Release Process

When a new version is released:
1. Towncrier collects all fragment files
2. Generates release notes grouped by type (Added, Changed, Fixed, Removed)
3. Updates `CHANGELOG.md` with the new version section
4. Deletes the fragment files from this directory

## Keep a Changelog Format

Fragments follow the [Keep a Changelog](https://keepachangelog.com/) format with these categories:

- **Added** for new features
- **Changed** for changes in existing functionality
- **Fixed** for bug fixes
- **Removed** for removed features
