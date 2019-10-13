mod builder;
mod components;

pub use builder::ProgramBuilder;

use std::collections::HashMap;
use std::fmt::{self, Display};
use z3::ast::{Ast, Bool, Dynamic, Int, BV as BitVec};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub usize);

impl Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

pub enum Operator {
    // Declare a new variable.
    Var,

    // Unary operators.
    Const(i32),
    Eqz(Id),
    Clz(Id),
    Ctz(Id),
    Popcnt(Id),

    // Binary relations.
    Eq(Id, Id),
    Ne(Id, Id),
    LtS(Id, Id),
    LtU(Id, Id),
    GtS(Id, Id),
    GtU(Id, Id),
    LeS(Id, Id),
    LeU(Id, Id),
    GeS(Id, Id),
    GeU(Id, Id),

    // Binary arithmetic.
    Add(Id, Id),
    Sub(Id, Id),
    Mul(Id, Id),
    DivS(Id, Id),
    DivU(Id, Id),
    RemS(Id, Id),
    RemU(Id, Id),
    And(Id, Id),
    Or(Id, Id),
    Xor(Id, Id),
    Shl(Id, Id),
    ShrS(Id, Id),
    ShrU(Id, Id),
    Rotl(Id, Id),
    Rotr(Id, Id),
}

impl Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operator::Var => write!(f, "var"),
            Operator::Const(c) => write!(f, "const {}", c),
            Operator::Eqz(id) => write!(f, "eqz {}", id),
            Operator::Clz(id) => write!(f, "clz {}", id),
            Operator::Ctz(id) => write!(f, "ctz {}", id),
            Operator::Popcnt(id) => write!(f, "popcnt {}", id),
            Operator::Eq(a, b) => write!(f, "eq {}, {}", a, b),
            Operator::Ne(a, b) => write!(f, "ne {}, {}", a, b),
            Operator::LtS(a, b) => write!(f, "lts {}, {}", a, b),
            Operator::LtU(a, b) => write!(f, "ltu {}, {}", a, b),
            Operator::GtS(a, b) => write!(f, "gts {}, {}", a, b),
            Operator::GtU(a, b) => write!(f, "gtu {}, {}", a, b),
            Operator::LeS(a, b) => write!(f, "les {}, {}", a, b),
            Operator::LeU(a, b) => write!(f, "leu {}, {}", a, b),
            Operator::GeS(a, b) => write!(f, "ges {}, {}", a, b),
            Operator::GeU(a, b) => write!(f, "geu {}, {}", a, b),
            Operator::Add(a, b) => write!(f, "add {}, {}", a, b),
            Operator::Sub(a, b) => write!(f, "sub {}, {}", a, b),
            Operator::Mul(a, b) => write!(f, "mul {}, {}", a, b),
            Operator::DivS(a, b) => write!(f, "divs {}, {}", a, b),
            Operator::DivU(a, b) => write!(f, "divu {}, {}", a, b),
            Operator::RemS(a, b) => write!(f, "rems {}, {}", a, b),
            Operator::RemU(a, b) => write!(f, "remu {}, {}", a, b),
            Operator::And(a, b) => write!(f, "and {}, {}", a, b),
            Operator::Or(a, b) => write!(f, "or {}, {}", a, b),
            Operator::Xor(a, b) => write!(f, "xor {}, {}", a, b),
            Operator::Shl(a, b) => write!(f, "shl {}, {}", a, b),
            Operator::ShrS(a, b) => write!(f, "shrs {}, {}", a, b),
            Operator::ShrU(a, b) => write!(f, "shru {}, {}", a, b),
            Operator::Rotl(a, b) => write!(f, "rotl {}, {}", a, b),
            Operator::Rotr(a, b) => write!(f, "rotr {}, {}", a, b),
        }
    }
}

pub struct Instruction {
    pub result: Id,
    pub operator: Operator,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}", self.result, self.operator)
    }
}

pub struct Program {
    pub instructions: Vec<Instruction>,
}

impl Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in &self.instructions {
            write!(f, "{}\n", i)?;
        }
        Ok(())
    }
}

impl Program {
    pub fn synthesize(
        context: &z3::Context,
        spec: &Specification,
        library: &Library,
    ) -> Result<Program, ()> {
        assert!(!library.components.is_empty());

        let params: Vec<_> = library
            .components
            .iter()
            .flat_map(|c| c.spec.inputs.iter().cloned())
            .collect();

        let results: Vec<_> = library
            .components
            .iter()
            .map(|c| c.spec.output.clone())
            .collect();

        let locations: HashMap<_, _> = params
            .iter()
            .chain(&results)
            .chain(&spec.inputs)
            .chain(Some(&spec.output))
            .cloned()
            .map(|b| (b, Int::fresh_const(context, "location")))
            .collect();

        let wfp = Self::well_formed_program(
            context,
            &spec.inputs,
            &library.components,
            &params,
            &results,
            &locations,
        );

        let conn = Self::connections(context, &locations);

        let lib = library
            .components
            .iter()
            .fold(Bool::from_bool(context, true), |lib, c| {
                lib.and(&[&c.spec.expression])
            });

        let existentials: Vec<Dynamic> = locations.values().cloned().map(|v| v.into()).collect();
        let existentials: Vec<&_> = existentials.iter().collect();

        let universals: Vec<Dynamic> = spec
            .inputs
            .iter()
            .chain(Some(&spec.output))
            .chain(&params)
            .chain(&results)
            .cloned()
            .map(|b| b.into())
            .collect();
        let universals: Vec<&_> = universals.iter().collect();

        let synthesis = z3::ast::exists_const(
            context,
            &existentials,
            &[],
            &z3::ast::forall_const(
                context,
                &universals,
                &[],
                &wfp.and(&[&lib.and(&[&conn]).implies(&spec.expression)])
                    .into(),
            ),
        );

        let solver = z3::Solver::new(context);
        solver.assert(&synthesis.as_bool().unwrap());
        match solver.check() {
            z3::SatResult::Unknown => eprintln!("unknown"),
            z3::SatResult::Unsat => eprintln!("unsat"),
            z3::SatResult::Sat => eprintln!("sat"),
        }
        let model = solver.get_model();
        for c in &library.components {
            for i in &c.spec.inputs {
                dbg!(i);
                dbg!(&locations[i]);
                dbg!(model.eval(&locations[i]));
            }
            dbg!(&c.spec.output);
            dbg!(&locations[&c.spec.output]);
            dbg!(model.eval(&locations[&c.spec.output]));
        }

        unimplemented!()
    }

    fn well_formed_program<'a>(
        context: &'a z3::Context,
        inputs: &[BitVec<'a>],
        components: &[Component<'a>],
        params: &[BitVec<'a>],
        results: &[BitVec<'a>],
        locations: &HashMap<BitVec<'a>, Int<'a>>,
    ) -> Bool<'a> {
        let mut wfp = Vec::with_capacity(
            // Acyclic and consistent.
            2 +
                // Lower and upper bounds on params.
                params.len() * 2 +
                // Lower and upper bounds on results.
                results.len() * 2,
        );

        wfp.push(Self::consistent(context, results, locations));
        wfp.push(Self::acyclic(context, components, locations));

        let i_len = Int::from_i64(context, inputs.len() as i64);
        let m = Int::from_i64(context, (components.len() + inputs.len()) as i64);
        let zero = Int::from_i64(context, 0);

        for p in params {
            let l = &locations[p];
            // 0 <= l
            wfp.push(zero.le(l));
            // l < M
            wfp.push(l.lt(&m));
        }

        for r in results {
            let l = &locations[r];
            // |i| <= l
            wfp.push(i_len.le(l));
            // l < m
            wfp.push(l.lt(&m));
        }

        let wfp: Vec<&_> = wfp.iter().collect();
        Bool::from_bool(context, true).and(&wfp)
    }

    fn consistent<'a>(
        context: &'a z3::Context,
        results: &[BitVec<'a>],
        locations: &HashMap<BitVec<'a>, Int<'a>>,
    ) -> Bool<'a> {
        use itertools::Itertools;
        results
            .iter()
            .tuple_combinations()
            .fold(Bool::from_bool(context, true), |cons, (x, y)| {
                let x = &locations[x];
                let y = &locations[y];
                cons.and(&[&x._eq(y).not()])
            })
    }

    fn acyclic<'a>(
        context: &'a z3::Context,
        components: &[Component<'a>],
        locations: &HashMap<BitVec<'a>, Int<'a>>,
    ) -> Bool<'a> {
        let mut acycs = Vec::with_capacity(components.len() * 2);
        for c in components {
            let output_location = &locations[&c.spec.output];
            for i in &c.spec.inputs {
                let input_location = &locations[i];
                acycs.push(input_location.lt(&output_location));
            }
        }
        let acycs: Vec<&_> = acycs.iter().collect();
        Bool::from_bool(context, true).and(&acycs)
    }

    fn connections<'a>(
        context: &'a z3::Context,
        locations: &HashMap<BitVec<'a>, Int<'a>>,
    ) -> Bool<'a> {
        let mut conn = Vec::with_capacity(locations.len() * locations.len() / 2);
        let locs: Vec<(_, _)> = locations.iter().collect();
        for (i, (x, l_x)) in locs.iter().enumerate() {
            for (y, l_y) in &locs[i + 1..] {
                conn.push(l_x._eq(l_y).implies(&x._eq(y)));
            }
        }
        let conn: Vec<&_> = conn.iter().collect();
        Bool::from_bool(context, true).and(&conn)
    }
}

pub struct Specification<'a> {
    inputs: Vec<BitVec<'a>>,
    output: BitVec<'a>,
    expression: Bool<'a>,
}

pub struct Library<'a> {
    pub components: Vec<Component<'a>>,
}

pub struct Component<'a> {
    operator: Operator,
    spec: Specification<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_id() {
        assert_eq!(Id(42).to_string(), "%42");
    }

    #[test]
    fn display_operator() {
        assert_eq!(Operator::Mul(Id(1), Id(2)).to_string(), "mul %1, %2");
    }

    #[test]
    fn display_instruction() {
        let instr = Instruction {
            result: Id(3),
            operator: Operator::And(Id(1), Id(2)),
        };
        assert_eq!(instr.to_string(), "%3 = and %1, %2");
    }

    #[test]
    fn display_program() {
        let mut builder = ProgramBuilder::new();
        let a = builder.var();
        let b = builder.var();
        let c = builder.const_(1);
        let d = builder.add(a, c);
        let _e = builder.and(b, d);
        let program = builder.finish();
        assert_eq!(
            program.to_string(),
            "%0 = var\n\
             %1 = var\n\
             %2 = const 1\n\
             %3 = add %0, %2\n\
             %4 = and %1, %3\n\
             "
        );
    }

    #[test]
    fn synthesize() {
        let mut config = z3::Config::new();
        config.set_model_generation(true);

        let context = z3::Context::new(&config);

        let library = Library {
            components: vec![Component::mul(&context), Component::const_(&context)],
        };

        let inputs = vec![BitVec::new_const(&context, "x", 32)];
        let output = BitVec::new_const(&context, "two_x", 32);
        let expression = inputs[0].bvadd(&inputs[0])._eq(&output);
        let spec = Specification {
            inputs,
            output,
            expression,
        };

        let p = Program::synthesize(&context, &spec, &library);
    }
}
