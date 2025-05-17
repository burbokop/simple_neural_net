mod arr;
mod layer;
mod perceptron;

pub use {arr::*, layer::*, perceptron::*};

pub use simple_neural_net_macro::compose_layers;

pub mod normalizers {

    /// Normalizes to value from -1 to 1
    #[inline(always)]
    pub fn sigmoid(x: f64) -> f64 {
        use std::f64::consts::E;
        2. / (1. + E.powf(-x)) - 1.
    }

    /// Normalizes to value from -1 to 1
    #[inline(always)]
    pub fn fast_fake_sigmoid(x: f64) -> f64 {
        x / (1. + x.abs())
    }

    #[inline(always)]
    pub fn identity<T>(x: T) -> T {
        x
    }

    #[inline(always)]
    pub fn sigmoid_derivative(y: f64) -> f64 {
        2. * y * (1. - y)
    }
}
