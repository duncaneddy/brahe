# Design Philosophy & Decisions

This page documents significant design decisions of the library. Right now this is mostly just what I think is "good design" for an astrodynamics library, which is highly subjective and somewhat hard to write down concretely. Over time I hope to expand this page to include more specific design decisions made in the library, along with the reasoning behind them.

## On Development Philosophy

I think the [rv](https://github.com/promised-ai/rv?tab=readme-ov-file#contributing) contributing guidelines express my philosophy well, so I am copying them here with minor modifications.

Brahe originated out of a frustration with existing astrodynamics software. I wanted something that was easy to use, easy to install, and easy to extend. I wanted something that could help me solve the problems I cared about quickly. I wanted it to be succinct yet expressive, easy-to-use, yet performant. I wanted it to be fun. This is something I've been working on for over a decade of my life in one for or another, with all the learnings and opinions that come with that experience.

All of this is to say, the ultimate goal of Brahe is to solve the problems I care about. If that means that others find it useful—that's great! I'd love that it can also be useful to others and help them avoid the frustrations and problems I've had. I will gladly accept contributions that align with my vision and help improve Brahe in ways that I find valuable, but for now I reserve the right to steer the project in the direction that I find most compelling. Depending on how things go, am open to opening-up more decision-making to the community in the future.

If you want to take things in a different direction, that's great! I encourage you to fork the project and build your own version that suits your needs. Part of my vision for Brahe is that it should be easy to extend and modify so that I'm not a single point of failure for the project.

## Rust Core with Python Bindings

The core of the library is implemented in Rust for performance and safety, with Python bindings provided for ease of use and accessibility to the Python community. This design choice allows us to leverage Rust's strengths while still providing a user-friendly interface in Python. Providing Python bindings also opens the library to a wider audience, as Python is a popular language in the scientific and engineering communities.

## Tightly Coupled Documentation

The documentation is designed to be tightly coupled with the codebase, ensuring that users have access to up-to-date and relevant information. This is achieved by organizing the documentation in a way that mirrors the structure of the code, and automatically testing code examples within the documentation to ensure sure that documentation cannot be released without it working. This approach helps maintain consistency between the library's functionality and its documentation, making it easier for users to understand and utilize the library effectively.

## Earth-Centered Focus

Currently the library is focused on Earth-centered applications, while extending to other celestial bodies is not ruled out in the future, they are not a design priority. This choice has enabled us to optimize the design of the library API by dropping support for multiple central bodies, simplifying the user experience for the primary use case.

## Do the Rightest Thing

When faced with design decisions, I try to choose the option that feels like the "rightest" thing to do. This is generally what is the most common, most user-fiendly, or most extensible option. The goal is to make the library as easy to use and understand as possible, while still providing the necessary functionality and flexibility for advanced users.

For example, when designing the API for propagators, we've taken an Earth-centered approach, simplifying the interface for the most common use case. It doesn't preclude future support for other central bodies, but it does make the main library easier to get up an running with for the majority of users.

Another example is "ECI" and "ECEF" frame naming. While these are not strictly correct terms (they should be "TEME" and "ITRF"), they are much commonly used in the astrodynamics community, and so we've chosen to use them in the library to make it easier for users to understand and work with. They're currently backed by the GCRF and ITRF frames currently, but having the main functions use the common terminology makes it easier for users to get started with the library.

## S-Type and D-Type Structs

In Brahe, we distinguish between "S-Type" (Static) and "D-Type" (Dynamic) structs to clarify their intended usage and mutability. This is partially a knock-on effect from nalgebra's design and naming, but also reflects the reality of stack and heap allocation in programming. Brahe exposes both S-Type and D-Type structs, traits, and functions in Rust. In python however, only D-Type structs and functions are exposed, as Python's dynamic nature makes the distinction less relevant. By only having D-Type structs in Python, we simplify the API and make it easier for users to compose and manipulate objects without worrying about the underlying S-Type and D-Type distinctions.
