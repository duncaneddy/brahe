# Contribution

We welcome contributions from external users, individuals, and organizations to the project. If you are interested in starting contributing 

## Pull Request Template

The project recommends using 

```markdown
## Summary

{{ Provide a clear summary of the problem this pull request addresses }}

### Attention Areas

{{ Optional Section: Call out any specific areas the reviewers should pay attention to. Any contentious or critical design and implementation decisions should be listed here}}

1. This changes how the entire package works
2. This CR solves ABC by doing DEF, but that might affect XYZ

----

## Change Description

{{ Provide a bulletted listed of the items changed }}

1. This pull request template

----

## Testing

{{ How was this pull request tested? }}

- [ ] Test suite passes
- [ ] Unit tests addded
- [ ] No testing performed 

----

## Revisions

{{ List modifications to this pull request between revisions. First revision can simply be initial commit }}

Revision 1:

- Initial submission

----

## Related Issues

{{ Optional Section, Link any related issues }}

- Issue 1: 

```

## Rust Docstring Template

New functions implemented in rust are expected to use the following docstring to standardize information on functions to
enable users to more easily navigate and learn the library.

```markdown
{{ Function Description }}

## Arguments

* `argument_name`: {{ Arugment description}}. Units: {{ Optional, Units as (value). e.g. (rad) or (deg)}}

## Returns

* `value_name`: {{ Value description}}. Units: {{ Optional, Units as (value). e.g. (rad) or (deg)}}

## Examples
\`\`\`rust
{{ Implement shor function in language }}
\`\`\`

## References:
1. {{ author, *title/journal*, pp. page_number, eq. equation_number, year}}
2. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 24, eq. 2.43 & 2.44, 2012.
```