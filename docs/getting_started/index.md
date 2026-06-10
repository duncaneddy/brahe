# Getting Started

This section provides a quick introduction to using Brahe and the main concepts you need to understand to get started.

The documentation is organized into the following sections:

- [Installation](../installation.md): Instructions for installing Brahe and its dependencies. If you havne't installed Brahe yet, **start here**.
- [User Guide](../learn/index.md): The user guide provides more comprehensive module-by-module documentation covering the capabilities each module provides with examples on how to use it.
- [Python API Reference](../library_api/index.md): This is the Python API reference documentation, which provides detailed information on all public classes, functions, and methods available in Brahe's Python API. It is organized by module and is automatically generated from source code docstrings. If you are looking for specific information on how to use a particular function or class, this is the place to find it.
- [Rust API Reference](https://docs.rs/brahe/latest/brahe/): This is the Rust API reference documentation, which provides the same detailed information as the Python API reference but for Brahe's Rust API. It is also organized by module and automatically generated from source code docstrings. It uses the standard Crate API reference format common in Rust documentation.
- [Examples](../examples/index.md): This section contains a collection of worked examples that demonstrate how to use Brahe to solve various problems or accomplish specific tasks. The examples are designed to be practical and cover a range of use cases, from basic to more advanced.
- [Getting Started](index.md): This section! The getting started guide provides a high-level overview of the main concepts and components of Brahe, along with a quick introduction to using it. It is designed to help new users get up to speed quickly and understand the core ideas behind Brahe before diving into the more detailed documentation in the other sections. If you are new to Brahe, this is a great place to start!

!!! tip "Compiling Code Examples"
    In the brahe documentation, all code examples are provided excerpts from full working code examples that are tested as part of the documentation build and release process. For this reason, all code examples in the documentation are guaranteed to compile and run correctly with the latest version of Brahe.

## Installation

For most projects you can quickly install and start using Brahe using language-native package managers. Brahe is available in Rust and Python. Choose the installation method that best fits your project:

=== "Python"

    ``` bash
    # Using uv (preferred)
    uv add brahe

    # Or using pip
    pip install brahe

    # Or with optional plot dependencies
    uv add "brahe[plots]"
    pip install "brahe[plots]"
    ```

=== "Rust"

    ``` bash
    crago add brahe
    ```

For full installation instructions, including building from source for developopment and modifications, see the [Installation Guide](../installation.md).

## Starting Your Journey

Now that you've gotten brahe installed, you can start using it! Head over to the [First Script](first_script.md) page to write your first Brahe script and get a quick introduction to using the library.

As you continue to use this guide you can select the language you want to see the code examples in using the language selector at the top of each section. If you want to see the expected output from running the examples, you can click the "Output" tab at the end of each example to see the expected output from running the code examples.

=== "Python"

    ``` python
    print("Hello, Brahe!")
    ```

=== "Rust"

    ``` rust
    fn main() {
        println!("Hello, Brahe!");
    }
    ```

??? example "Output"
    === "Python"
        ``` bash
        Hello, Brahe!
        ```

    === "Rust"
        ```
        Hello, Brahe!
        ```