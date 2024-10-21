use proc_macro::TokenStream;
use syn::{self, parse::Parse, Ident, LitInt, Token};
use quote::{format_ident, quote, ToTokens};

#[derive(Debug)]
struct ParsedInput {
    type_name: Ident,
    layers: Vec<usize>,
}

impl Parse for ParsedInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            panic!("At least a type must be specified");
        }
        let type_name = input.parse::<Ident>()?;
        let mut layers: Vec<usize> = Default::default();
        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            layers.push(input.parse::<LitInt>()?.base10_parse()?);
        }
        Ok(Self { type_name, layers })
    }
}

#[derive(Debug, Clone)]
struct PerceptronLayerType {
    input_size: usize,
    output_size: usize,
}

impl ToTokens for PerceptronLayerType {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let input_size = self.input_size;
        let output_size = self.output_size;
        tokens.extend(quote!(simple_neural_net::PerceptronLayer<T, #input_size, #output_size>))
    }
}

#[derive(Debug)]
struct PerceptronLayerField {
    index: usize,
    tp: PerceptronLayerType,
}

impl ToTokens for PerceptronLayerField {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let field_name = format_ident!("l{}", self.index);
        let tp = &self.tp;
        tokens.extend(quote!(#field_name: #tp))
    }
}

struct InitFromTupleField {
    index: usize,
}

impl ToTokens for InitFromTupleField {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let field_name = format_ident!("l{}", self.index);
        let tuple_field_name = syn::Index::from(self.index);
        tokens.extend(quote!(#field_name: value.#tuple_field_name))
    }
}

#[derive(Debug)]
struct ProceedCall {
    layer_index: usize,
    input_index: usize,
    output_input: usize,
}

impl ToTokens for ProceedCall {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let layer_name = format_ident!("l{}", self.layer_index);
        let input_name = format_ident!("r{}", self.input_index);
        let output_name = format_ident!("r{}", self.output_input);
        tokens.extend(quote!(let #output_name = self.#layer_name.proceed(&#input_name, normalizer)))
    }
}

#[proc_macro]
pub fn compose_layers(input: TokenStream) -> TokenStream {
    let parsed = syn::parse_macro_input!(input as ParsedInput);
    if parsed.layers.len() < 2 {
        panic!("Layers count must be at leas 2");
    }

    let fields = {
        let mut layers: Vec<PerceptronLayerField> = Default::default();
        layers.reserve(parsed.layers.len() - 1);
        for i in 0..parsed.layers.len() - 1 {
            let input_size = parsed.layers[i + 0];
            let output_size = parsed.layers[i + 1];

            layers.push(PerceptronLayerField { index: i, tp: PerceptronLayerType { input_size, output_size } });
        }
        layers
    };

    let proceed_calls = {
        let mut calls: Vec<ProceedCall> = Default::default();
        calls.reserve(parsed.layers.len() - 1);
        for i in 1..parsed.layers.len() - 1 {
            calls.push(ProceedCall { layer_index: i, input_index: i - 1, output_input: i });
        }
        calls
    };

    let type_name = parsed.type_name;
    let input_size = parsed.layers[0];
    let output_size = parsed.layers[parsed.layers.len() - 1];
    let init_args: Vec<_> = (0..parsed.layers.len() - 1).map(|x| format_ident!("l{}", x)).collect();
    let final_result_var_name = format_ident!("r{}", proceed_calls.len());
    let layer_types: Vec<_> = fields.iter().map(|x|x.tp.clone()).collect();
    let init_from_tuple_field: Vec<_> = (0..parsed.layers.len() - 1).map(|index| InitFromTupleField{ index }).collect();

    quote!(
        #[derive(Debug, serde::Serialize, serde::Deserialize)]
        struct #type_name<T> {
            #(#fields,)*
        }

        impl<T> #type_name<T> {
            #[inline(always)]
            pub fn new(#(#fields,)*) -> Self {
                Self { #(#init_args,)* }
            }
        }

        impl<T> From<(#(#layer_types,)*)> for #type_name<T> {
            #[inline(always)]
            fn from(value: (#(#layer_types,)*)) -> Self {
                Self { #(#init_from_tuple_field,)* }
            }
        }

        impl<T> simple_neural_net::Layer<T, #input_size, #output_size> for #type_name<T>
        where
            T: Clone,
            T: std::ops::Mul<Output = T>,
            T: std::ops::Sub<Output = T>,
            T: std::iter::Sum<T>,
        {
            #[inline(always)]
            fn proceed(&self, input: &[T; #input_size], normalizer: fn(T) -> T) -> simple_neural_net::Arr<T, #output_size> {
                let r0 = self.l0.proceed(input, normalizer);
                #(#proceed_calls;)*
                #final_result_var_name
            }
        }
    ).into()
}
