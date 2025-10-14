# Design Decisions

This page documents significant design decisions of the library.

## Default 64-bit Floating Point Functions

There is a decision to be made as to whether to support both 32-bit and 64-bit
operations for all standard functions that could be used with either type. However,
this results in duplicating a large amount of code as Rust does not support overloading primitive types.
While some space missions may use lower-precision processors that do not support 64-bit floating point operations,
the majority of modern computers and processors do support 64-bit floating point operations. Furthermore,
the primary focus of this library is for us in terrestrial applications, and analysis, or space-processors that
do support 64-bit floating point operations.

Therefore, the decision is to only support 64-bit floating point operations for the time being.

## Inline PyO3 Annotations vs Written Wrappers

There is a decision to use PyO3 macros to generate Python bindings or 
to write them manually. Currently, the decision is to write them manually
to have more control over the generated code and be able to more easily
navigate issues related to type conversions that can arise. As a secondary
benefit separate python-formatted docstrings can be written for the new functions.

PyO3 macros would ultimately be preferable to reduce duplication of code, the 
amount of boilerplate, and to make the code more maintainable. However, at
the current time the complexity and difficulty to get them working is not
worth the effort. This decision may be revisited in the future, and any change 
proposals and pull requests to use PyO3 macros are welcome.