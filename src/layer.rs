use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

use crate::{Arr, Perceptron};
use serde::{Deserialize, Serialize};

pub trait Layer<T, const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    fn proceed(&self, input: &[T; INPUT_SIZE], normalizer: fn(T) -> T) -> Arr<T, OUTPUT_SIZE>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerceptronLayer<T, const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    perceptrons: Arr<Perceptron<T, INPUT_SIZE>, OUTPUT_SIZE>,
}

impl<T, const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> PerceptronLayer<T, INPUT_SIZE, OUTPUT_SIZE> {
    pub fn perceptrons(&self) -> &Arr<Perceptron<T, INPUT_SIZE>, OUTPUT_SIZE> {
        &self.perceptrons
    }
}

impl<T, const INPUT_SIZE: usize, const OUTPUT_SIZE: usize>
    From<[Perceptron<T, INPUT_SIZE>; OUTPUT_SIZE]> for PerceptronLayer<T, INPUT_SIZE, OUTPUT_SIZE>
{
    #[inline(always)]
    fn from(value: [Perceptron<T, INPUT_SIZE>; OUTPUT_SIZE]) -> Self {
        Self {
            perceptrons: Arr::from(value),
        }
    }
}

impl<T, const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Layer<T, INPUT_SIZE, OUTPUT_SIZE>
    for PerceptronLayer<T, INPUT_SIZE, OUTPUT_SIZE>
where
    T: Clone,
    T: Mul<Output = T>,
    T: Sub<Output = T>,
    T: Sum<T>,
{
    #[inline(always)]
    fn proceed(&self, input: &[T; INPUT_SIZE], normalizer: fn(T) -> T) -> Arr<T, OUTPUT_SIZE> {
        unsafe {
            let mut result: Arr<T, OUTPUT_SIZE> = Arr::uninitialized();
            for i in 0..OUTPUT_SIZE {
                result[i] = self.perceptrons[i].proceed(input, normalizer)
            }
            result
        }
    }
}
