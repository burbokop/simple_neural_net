use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

use super::Arr;

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Perceptron<T, const SIZE: usize> {
    weights: Arr<T, SIZE>,
    bias: T,
}

impl<const SIZE: usize> Default for Perceptron<f64, SIZE> {
    #[inline(always)]
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: Arr::rand(&mut rng, (-1.)..=(1.)),
            bias: rng.gen_range((-1.)..=(1.)),
        }
    }
}

impl<T, const SIZE: usize> From<([T; SIZE], T)> for Perceptron<T, SIZE> {
    #[inline(always)]
    fn from(value: ([T; SIZE], T)) -> Self {
        Self {
            weights: Arr::from(value.0),
            bias: value.1,
        }
    }
}

impl<T, const SIZE: usize> Perceptron<T, SIZE> {
    #[inline(always)]
    pub fn new(weights: Arr<T, SIZE>, bias: T) -> Self {
        Self { weights, bias }
    }

    #[inline(always)]
    pub fn bias(&self) -> &T {
        &self.bias
    }

    #[inline(always)]
    pub fn weights(&self) -> &Arr<T, SIZE> {
        &self.weights
    }

    #[inline(always)]
    pub fn bias_mut(&mut self) -> &mut T {
        &mut self.bias
    }

    #[inline(always)]
    pub fn weights_mut(&mut self) -> &mut Arr<T, SIZE> {
        &mut self.weights
    }

    #[inline(always)]
    pub fn proceed(&self, activations: &[T; SIZE], normalizer: fn(T) -> T) -> T
    where
        T: Clone,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
        T: Sum<T>,
    {
        normalizer(
            activations
                .into_iter()
                .cloned()
                .zip(self.weights.iter().cloned())
                .map(|(a, b)| a * b)
                .sum::<T>()
                - self.bias.clone(),
        )
    }
}
