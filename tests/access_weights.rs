use simple_neural_net::{compose_layers, Layer};

#[test]
fn access_weights() {
    compose_layers!(Net, 4, 3, 2, 1);

    let net = Net::new(
        [
            ([0., 1., 2., 3.], 12.).into(),
            ([4., 5., 6., 7.], 13.).into(),
            ([8., 9., 10., 11.], 13.).into(),
        ]
        .into(),
        [([14., 15., 16.], 20.).into(), ([17., 18., 19.], 21.).into()].into(),
        [([22., 23.], 24.).into()].into(),
    );

    assert_eq!(net.l0.perceptrons()[0].weights()[0], 0.);
    assert_eq!(*net.l0.perceptrons()[0].bias(), 12.);
}
