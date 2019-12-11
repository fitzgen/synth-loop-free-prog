use synth_loop_free_prog::*;

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

    let mut config = z3::Config::new();
    config.set_bool_param_value("auto_config", false);
    config.set_model_generation(true);

    let context = z3::Context::new(&config);

    let problems: Vec<(&'static str, fn(&z3::Context) -> Program)> = benchmarks! {
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        // p23,
    };

    for (name, p) in problems {
        println!("==================== {} ====================", name);
        let then = std::time::Instant::now();
        let mut program = p(&context);
        let elapsed = then.elapsed();

        println!(
            "\nElapsed: {}.{:03}s\n",
            elapsed.as_secs(),
            elapsed.subsec_millis()
        );
        println!("Synthesized:\n\n{}", program);
        program.dce();
        println!("DCE'd:\n\n{}", program);
    }
}

fn p1(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let _ = builder.and(a, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p2(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.add(a, b);
    let _ = builder.and(a, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p3(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(0);
    let c = builder.sub(b, a);
    let _ = builder.and(a, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p4(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let _ = builder.xor(a, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p5(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let _ = builder.or(a, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p6(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.add(a, b);
    let _ = builder.or(a, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p7(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(0xffff_ffff_ffff_ffff);
    let c = builder.xor(a, b);
    let d = builder.const_(1);
    let e = builder.add(c, d);
    let _ = builder.and(c, e);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p8(context: &z3::Context) -> Program {
    let library = Library::brahma_std();

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(1);
    let c = builder.sub(a, b);
    let d = builder.const_(0xffff_ffff_ffff_ffff);
    let e = builder.xor(a, d);
    let _ = builder.and(c, e);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p9(context: &z3::Context) -> Program {
    let mut library = Library::brahma_std();
    library.components.push(component::const_(Some(31)));

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let b = builder.const_(31);
    let c = builder.shru(a, b);
    let d = builder.xor(a, c);
    let _ = builder.sub(d, c);
    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}

fn p23(context: &z3::Context) -> Program {
    let library = Library {
        components: vec![
            component::const_(Some(1)),
            component::shr_u(),
            component::const_(Some(0x5555_5555_5555_5555)),
            component::and(),
            component::sub(),
            component::const_(Some(0x3333_3333_3333_3333)),
            component::and(),
            component::const_(Some(2)),
            component::shr_u(),
            component::const_(Some(0x3333_3333_3333_3333)),
            component::and(),
            component::add(),
            component::const_(Some(4)),
            component::shr_u(),
            component::add(),
            component::const_(Some(0x0f0f_0f0f_0f0f_0f0f)),
            component::and(),
        ],
    };

    let mut builder = ProgramBuilder::new();
    let a = builder.var();
    let _ = builder.popcnt(a);

    let spec = builder.finish();

    Program::synthesize(&context, &spec, &library).unwrap()
}
