# Your First Brahe Script

This page will walk you through setting up your environment and writing your first Brahe script. All subsequent examples in the getting started guide and documentation will assume you are using this same environment setup.

## Environment Setup

### Python

For python, we will assume you are using [uv](https://astral.sh/uv/) as your package manager and environment tool. If you haven't already installed it, you can so from their instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

### Rust

In rust we'll use [rust-script](https://github.com/fornwall/rust-script) to run a single-file rust script without needing to setup a full cargo project. You can install rust-script with cargo:

``` bash
cargo install rust-script
```

!!! note "Migration to cargo-script"
    Rust is building native support for running single-file scripts with "cargo script" in an upcoming release ([tracking issue](https://github.com/rust-lang/cargo/issues/12207)). Once this is available we will migrate over to using the native cargo support and remove the rust-script dependency.

## Preamble

To use Brahe in a script, we need to declare it as a dependency of the script so that it is properly imported when we run the script. To do this you can use the following preamble at the top of your script file:

=== "Python"

    ``` python
    # /// script
    # dependencies = ["brahe"]
    ```
=== "Rust"

    ``` rust
    //! ```cargo
    //! [dependencies]
    //! brahe = "*"
    //! nalgebra = { version = "*", features = ["serde-serialize"] }
    //! serde_json = "*"
    //! rayon = "*"
    //! ```
    ```

## Your First Script

Now we can write your first script. We'll use Brahe to predict the next time the International Space Station (ISS) will be in view of NASA Johnson Space Center (JSC) in Houston, TX.

To do this, we need to import brahe, download the latest ephemeris information for the ISS, and then use that information to predict the next pass of the ISS over JSC, and print the results.

To do this, add the following code to your script after the preamble:

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/first_script.py:4"
    ```

    Save this file as `first_script.py` and you can run it with:

    ``` bash
    uv run first_script.py
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/first_script.rs:4"
    ```

    Save this file as `first_script.rs` and you can run it with:

    ``` bash
    rust-script first_script.rs
    ```

You can see the expected output below:

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/first_script.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/first_script.rs.txt"
        ```
