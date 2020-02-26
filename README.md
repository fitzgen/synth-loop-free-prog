# `synth-loop-free-prog`

Implementing [*Synthesis of Loop-free Programs* by Gulwani et
al](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/pldi11-loopfree-synthesis.pdf)
in Rust, using the [Z3][] solver.

I explain the paper and walk through this implementation in my blog post
[*Synthesizing Loop-Free Programs with Rust and Z3*](https://fitzgeraldnick.com/2020/01/13/synthesizing-loop-free-programs.html).

## Example

```rust
use synth_loop_free_prog::*;

let mut builder = ProgramBuilder::new();
// ...build an unoptimized program...
let spec_program = builder.finish();

// Define a library of components that the synthesized program can use.
let library = Library {
    components: vec![
        component::add(),
        component::sub(),
        component::xor(),
        component::shl(),
        // etc...
    ],
};

let config = z3::Config::new();
let context = z3::Context::new(&config);

// Synthesize an optimized program!
let optimized_program = Synthesizer::new(&context, &library, &spec_program)
    // One hour timeout.
    .set_timeout(60 * 60 * 1000)
    // Synthesize optimally small programs.
    .should_synthesize_minimal_programs(true)
    // Start synthesis!
    .synthesize()?;

println!("Synthesized program:\n\n{}", optimized_program);
```

## Build

First, ensure that you have [Z3][] installed on your system:

```shell
# Something like this, depending on your OS.
$ sudo apt install libz3-dev
```

Then run

```shell
$ cargo build
```

## Testing

```shell
$ cargo test
```

## Running the Example Benchmarks

Run the all 25 benchmark programs from the paper (originally taken from
[Hacker's Delight](https://www.goodreads.com/book/show/276079.Hacker_s_Delight))
like this:

```shell
$ cargo run --example brahma
```

You can also run only the ones that finish pretty quickly like this:

```shell
$ cargo run --example brahma -- --only-fast
```

You can see a full listing of the available options with:

```shell
$ cargo run --example brahma -- --help
```

## Logging

Logging requires incoking `cargo` with `--features log` when building, running,
or testing.

At the `debug` log level, information about the progress of synthesis, the bit
width we're synthesizing at, and the example inputs is logged:

```shell
$ export RUST_LOG=synth_loop_free_prog=debug
```

At the `trace` level, every SMT query is additionally logged:

```shell
$ export RUST_LOG=synth_loop_free_prog=trace
```

[Z3]: https://github.com/Z3Prover/z3
