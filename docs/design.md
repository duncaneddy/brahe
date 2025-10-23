# Design Philosophy & Decisions

This page documents significant design decisions of the library. Right now this is mostly just what I think is "good design" for an astrodynamics library, which is highly subjective and somewhat hard to write down concretely. Over time I hope to expand this page to include more specific design decisions made in the library, along with the reasoning behind them.

## Rust Core with Python Bindings

The core of the library is implemented in Rust for performance and safety, with Python bindings provided for ease of use and accessibility to the Python community. This design choice allows us to leverage Rust's strengths while still providing a user-friendly interface in Python. Providing Python bindings also opens the library to a wider audience, as Python is a popular language in the scientific and engineering communities.

## Tightly Coupled Documentation

The documentation is designed to be tightly coupled with the codebase, ensuring that users have access to up-to-date and relevant information. This is achieved by organizing the documentation in a way that mirrors the structure of the code, and automatically testing code examples within the documentation to ensure sure that documentation cannot be released without it working. This approach helps maintain consistency between the library's functionality and its documentation, making it easier for users to understand and utilize the library effectively.

## Earth-Centered Focus

Currently the library is focused on Earth-centered applications, while extending to other celestial bodies is not ruled out in the future, they are not a design priority. This choice has enabled us to optimize the design of the library API by dropping support for multiple central bodies, simplifying the user experience for the primary use case.