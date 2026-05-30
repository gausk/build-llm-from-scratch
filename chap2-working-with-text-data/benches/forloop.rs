use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_for_loop(c: &mut Criterion) {
    c.bench_function("for_loop", |b| {
        b.iter(|| {
            let mut vec = vec![0usize; 10_000];

            for i in 1..=10000 {
                vec[i - 1] = black_box(i * i);
            }

            black_box(vec);
        })
    });
}

fn bench_for_loop_with_cap(c: &mut Criterion) {
    c.bench_function("for_loop_cap", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(10_000);

            for i in 1..=10000 {
                vec.push(black_box(i * i));
            }

            black_box(vec);
        })
    });
}

fn bench_iterator(c: &mut Criterion) {
    c.bench_function("iterator", |b| {
        b.iter(|| {
            let vec = (1..=10000)
                .map(|x| black_box(x * x))
                .collect::<Vec<usize>>();

            black_box(vec);
        })
    });
}

criterion_group!(
    benches,
    bench_for_loop,
    bench_iterator,
    bench_for_loop_with_cap
);
criterion_main!(benches);
