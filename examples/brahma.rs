use structopt::*;
use synth_loop_free_prog::{Result as SynthResult, *};

#[derive(StructOpt)]
struct Options {
    /// Set a timeout, in milliseconds.
    #[structopt(short = "t", long = "timeout")]
    timeout: Option<u32>,

    /// Synthesize the optimally smallest programs.
    #[structopt(short = "m", long = "minimal")]
    minimal: bool,

    /// Run only the problems that we can solver pretty fast.
    #[structopt(short = "f", long = "only-fast", conflicts_with = "problems")]
    only_fast: bool,

    /// When supplied, run only these problems instead of all problems.
    #[structopt(last = true)]
    problems: Vec<String>,
}

impl Options {
    fn should_run_problem(&self, problem: &str) -> bool {
        self.problems.is_empty() || self.problems.iter().position(|p| p == problem).is_some()
    }
}

macro_rules! benchmarks {
    ( $($name:ident,)* ) => {
        vec![
            $(
                (stringify!($name), $name as _),
            )*
        ]
    }
}

fn main() {
    env_logger::init();

    let mut opts = Options::from_args();
    if opts.only_fast {
        opts.problems = vec![
            "p1".to_string(),
            "p2".to_string(),
            "p3".to_string(),
            "p4".to_string(),
            "p5".to_string(),
            "p6".to_string(),
            "p7".to_string(),
        ];
    }

    let mut config = z3::Config::new();
    config.set_bool_param_value("auto_config", false);
    config.set_model_generation(true);

    let context = z3::Context::new(&config);

    let problems: Vec<(
        &'static str,
        fn(&z3::Context, &Options) -> SynthResult<Program>,
    )> = benchmarks! {
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p23,
    };

    for (name, p) in problems {
        if !opts.should_run_problem(name) {
            continue;
        }

        println!("==================== {} ====================", name);
        let then = std::time::Instant::now();
        let program = p(&context, &opts);
        let elapsed = then.elapsed();

        println!(
            "\nElapsed: {}.{:03}s\n",
            elapsed.as_secs(),
            elapsed.subsec_millis()
        );
        match program {
            Ok(mut prog) => {
                println!("Synthesized:\n\n{}", prog);
                prog.dce();
                println!("DCE'd:\n\n{}", prog);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}

fn synthesize(
    opts: &Options,
    context: &z3::Context,
    spec: &dyn Specification,
    library: &Library,
) -> SynthResult<Program> {
    Synthesizer::new(context, library, spec)?
        .set_timeout(opts.timeout)
        .should_synthesize_minimal_programs(opts.minimal)
        .synthesize()
}

fn p1(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let _ = builder.and(a, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p2(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.add(a, b);
    let _ = builder.and(a, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p3(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(0);
    let c = builder.sub(b, a);
    let _ = builder.and(a, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p4(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let _ = builder.xor(a, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p5(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let _ = builder.or(a, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p6(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.add(a, b);
    let _ = builder.or(a, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p7(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(0xffff_ffff_ffff_ffff);
    let c = builder.xor(a, b);
    let d = builder.const_(1);
    let e = builder.add(c, d);
    let _ = builder.and(c, e);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p8(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let d = builder.const_(0xffff_ffff_ffff_ffff);
    let e = builder.xor(a, d);
    let _ = builder.and(c, e);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p9(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let mut library = Library::brahma_std();
    library.components.push(component::const_(Some(31)));

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(31);
    let c = builder.shru(a, b);
    let d = builder.xor(a, c);
    let _ = builder.sub(d, c);
    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}

fn p23(context: &z3::Context, opts: &Options) -> SynthResult<Program> {
    let library = Library {
        components: vec![
            // a
            component::const_(Some(1)),
            component::shr_u(),
            // b
            component::const_(Some(0x5555_5555_5555_5555)),
            component::and(),
            // c
            component::sub(),
            // d
            component::const_(Some(0x3333_3333_3333_3333)),
            component::and(),
            // e
            component::const_(Some(2)),
            component::shr_u(),
            // f
            component::const_(Some(0x3333_3333_3333_3333)),
            component::and(),
            // g
            component::add(),
            // h
            component::const_(Some(4)),
            component::shr_u(),
            // i
            component::add(),
            // j
            component::const_(Some(0x0F0F_0F0F_0F0F_0F0F)),
            component::and(),
            // k
            component::const_(Some(8)),
            component::shr_u(),
            // l
            component::add(),
            // m
            component::const_(Some(16)),
            component::shr_u(),
            // n
            component::add(),
            // o
            component::const_(Some(0x3F)),
            component::and(),
        ],
    };

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let _ = builder.popcnt(a);

    let spec = builder.finish();

    synthesize(opts, context, &spec, &library)
}
