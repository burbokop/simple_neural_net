mod arr;
mod layer;
mod perceptron;

pub use {arr::*, layer::*, perceptron::*};

pub use simple_neural_net_macro::compose_layers;

pub mod normalizers {
    #[inline(always)]
    pub fn sigmoid(x: f64) -> f64 {
        use std::f64::consts::E;
        1. / (1. + E.powf(-x))
    }

    #[inline(always)]
    pub fn sigmoid_derivative(y: f64) -> f64 {
        y * (1. - y)
    }

    #[inline(always)]
    pub fn identity<T>(x: T) -> T {
        x
    }
}
