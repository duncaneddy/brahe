/*!
Defines the `NumericalIntegrator` trait and its implementations, which provides a common interface
for numerical integration routines.
*/

trait NumericalIntegrator {
    type Time;
    type State;
    type Derivative;

    // fn step();
    // fn varmat();


    // fn step<F>(
    //     &self,
    //     t: Self::Time,
    //     dt: Self::Time,
    //     state: Self::State,
    // ) -> Self::State
    // where
    //     F: Fn(Self::Time, Self::State) -> Self::Derivative;
    //
    // fn integrate<F>(
    //     &self,
    //     f: F,
    //     t: f64,
    //     dt: f64,
    //     state: Self::State,
    // ) -> Self::State
    // where
    //     F: Fn(Self::Time, Self::State) -> Self::Derivative;
}