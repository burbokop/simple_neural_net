use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use simple_neural_net::{normalizers, Arr, Layer as _, PerceptronLayer};

simple_neural_net::compose_layers!(Net, 16, 8, 8);

type Float = f64;

#[derive(Clone)]
pub struct Brain {
    net: Net<Float>,
}

pub(crate) struct VerboseOutput {
    pub output: Arr<Float, 8>,
    pub activations: ([Float; 16], [Float; 8], [Float; 8]),
}

impl Brain {
    pub fn layers(
        &self,
    ) -> (
        &PerceptronLayer<Float, 16, 8>,
        &PerceptronLayer<Float, 8, 8>,
    ) {
        (&self.net.l0, &self.net.l1)
    }

    pub(crate) fn new(weights_and_biases: [Float; 208]) -> Self {
        let l0w = &weights_and_biases[0..128];
        let l1w = &weights_and_biases[128..192];

        let l0b = &weights_and_biases[192..200];
        let l1b = &weights_and_biases[200..208];

        let net: Net<f64> = Net::new(
            [
                (l0w[000..016].try_into().unwrap(), l0b[0]).into(),
                (l0w[016..032].try_into().unwrap(), l0b[1]).into(),
                (l0w[032..048].try_into().unwrap(), l0b[2]).into(),
                (l0w[048..064].try_into().unwrap(), l0b[3]).into(),
                (l0w[064..080].try_into().unwrap(), l0b[4]).into(),
                (l0w[080..096].try_into().unwrap(), l0b[5]).into(),
                (l0w[096..112].try_into().unwrap(), l0b[6]).into(),
                (l0w[112..128].try_into().unwrap(), l0b[7]).into(),
            ]
            .into(),
            [
                (l1w[00..08].try_into().unwrap(), l1b[0]).into(),
                (l1w[08..16].try_into().unwrap(), l1b[1]).into(),
                (l1w[16..24].try_into().unwrap(), l1b[2]).into(),
                (l1w[24..32].try_into().unwrap(), l1b[3]).into(),
                (l1w[32..40].try_into().unwrap(), l1b[4]).into(),
                (l1w[40..48].try_into().unwrap(), l1b[5]).into(),
                (l1w[48..56].try_into().unwrap(), l1b[6]).into(),
                (l1w[56..64].try_into().unwrap(), l1b[7]).into(),
            ]
            .into(),
        );

        Brain { net }
    }

    pub(crate) fn proceed(&self, input: &[Float; 16]) -> Arr<Float, 8> {
        self.net.proceed(input, normalizers::fast_fake_sigmoid)
    }

    pub(crate) fn proceed_verbosely(&self, input: &[Float; 16]) -> VerboseOutput {
        let (r0, r1) = self
            .net
            .proceed_verbosely(input, normalizers::fast_fake_sigmoid);
        VerboseOutput {
            output: r1.clone(),
            activations: (input.clone(), *r0, *r1),
        }
    }
}

fn gen_array<const N: usize, R: Rng>(rng: &mut R) -> [Float; N] {
    let mut r: [Float; N] = [0.; N];
    for i in 0..N {
        r[i] = rng.gen();
    }
    r
}

fn main_benchmark(c: &mut Criterion) {
    let mut rng: Pcg64 = Seeder::from([0, 1, 2, 3, 4]).make_rng();
    let brain = Brain::new(gen_array(&mut rng));
    let input: [Float; 16] = rng.gen();

    c.bench_function("proceed", |b| {
        b.iter(|| black_box(brain.proceed(&input)[0]))
    });

    c.bench_function("proceed_verbosely", |b| {
        b.iter(|| black_box(brain.proceed_verbosely(&input).output[0]))
    });
}

criterion_group!(benches, main_benchmark);
criterion_main!(benches);
