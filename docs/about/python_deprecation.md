# Pure-Python Brahe Deprecation Notice

The older pure-Python version of brahe is currently being deprecated in favor of a mixed Rust-Python implementation, along with improved documentation. That means that the development on the `master` branch has been frozen and will no longer be developed against. Moving forward the `main` branch will be the primary branch for the project.

There will be point commits (less than `1.0.0`) during this period that include breaking changes.

Furthermore, initially the features of the new implementation will not be at partity with the old python implementation, so users should pin their requirements file to use the latest commit of the master branch:

```
brahe @ git+https://github.com/duncaneddy/brahe@master
```

To install and use the latest master branch via pip

```
pip install git+https://github.com/duncaneddy/brahe.git@master
```

The old master branch can be found [here](https://github.com/duncaneddy/brahe/tree/master).
