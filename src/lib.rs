#![deny(missing_debug_implementations)]

mod builder;
pub mod component;
mod operator;

pub use builder::ProgramBuilder;
pub use component::Component;
pub use operator::Operator;

use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use z3::ast::{Ast, Bool, Int, BV as BitVec};

fn fresh_immediate(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "immediate", 32)
}

fn fresh_param(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "param", 32)
}

fn fresh_result(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "result", 32)
}

fn fresh_input(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "input", 32)
}

fn fresh_output(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "output", 32)
}

#[derive(Debug)]
pub enum Error {
    NoComponents,
    SynthesisUnsatisfiable,
    SynthesisUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub usize);

impl Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

#[derive(Debug)]
pub struct Instruction {
    pub result: Id,
    pub operator: Operator,
}

impl Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} = {}", self.result, self.operator)
    }
}

pub struct Specification {
    arity: usize,
    make_expression: for<'a> fn(
        context: &'a z3::Context,
        inputs: &[BitVec<'a>],
        output: &BitVec<'a>,
    ) -> Bool<'a>,
}

impl fmt::Debug for Specification {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let Specification {
            ref arity,
            make_expression,
        } = self;
        f.debug_struct("Specification")
            .field("arity", arity)
            .field("make_expression", &(make_expression as *const _))
            .finish()
    }
}

/// A collection of components.
///
/// Multiple copies of a particular component may exist in the library, allowing
/// synthesis to use it multiple times.
#[derive(Debug)]
pub struct Library {
    pub components: Vec<Box<dyn Component>>,
}

impl Library {
    fn fresh_locations<'a>(&self, context: &'a z3::Context, num_inputs: usize) -> LocationVars<'a> {
        let inputs = (0..num_inputs)
            .map(|_| Int::fresh_const(context, "input_location"))
            .collect();
        let params = self
            .components
            .iter()
            .flat_map(|c| (0..c.arity()).map(|_| Int::fresh_const(context, "param_location")))
            .collect();
        let results = self
            .components
            .iter()
            .map(|_| Int::fresh_const(context, "result_location"))
            .collect();
        LocationVars {
            inputs,
            params,
            results,
        }
    }

    fn fresh_immediates<'a>(&self, context: &'a z3::Context) -> Vec<BitVec<'a>> {
        self.components
            .iter()
            .flat_map(|c| (0..c.immediates()).map(|_| fresh_immediate(context)))
            .collect()
    }

    fn fresh_param_vars<'a>(&self, context: &'a z3::Context) -> Vec<BitVec<'a>> {
        self.components
            .iter()
            .flat_map(|c| (0..c.arity()).map(|_| fresh_param(context)))
            .collect()
    }

    fn fresh_result_vars<'a>(&self, context: &'a z3::Context) -> Vec<BitVec<'a>> {
        self.components
            .iter()
            .map(|_| fresh_result(context))
            .collect()
    }
}

#[derive(Debug)]
struct LocationVars<'a> {
    inputs: Vec<Int<'a>>,
    // TODO: output: Int<'a>, ?
    params: Vec<Int<'a>>,
    results: Vec<Int<'a>>,
}

#[derive(Debug)]
struct Assignments {
    immediates: Vec<i32>,
    // The line in the program where the i^th input is defined (for all inputs
    // of all components).
    params: Vec<usize>,
    // The line in the program where the i^th component is located (and
    // therefore the i^th output is defined)..
    results: Vec<usize>,
}

impl Assignments {
    fn to_vars<'a>(
        &self,
        context: &'a z3::Context,
        num_inputs: usize,
    ) -> (LocationVars<'a>, Vec<BitVec<'a>>) {
        let inputs = (0..num_inputs)
            .map(|i| Int::from_i64(context, i as i64))
            .collect();
        let immediates = self
            .immediates
            .iter()
            .map(|imm| BitVec::from_i64(context, *imm as i64, 32))
            .collect();
        let params = self
            .params
            .iter()
            .map(|p| Int::from_i64(context, *p as i64))
            .collect();
        let results = self
            .results
            .iter()
            .map(|r| Int::from_i64(context, *r as i64))
            .collect();
        (
            LocationVars {
                inputs,
                params,
                results,
            },
            immediates,
        )
    }

    fn to_program(&self, num_inputs: usize, library: &Library) -> Program {
        let mut b = ProgramBuilder::new();
        for _ in 0..num_inputs {
            b.var();
        }
        let mut program = b.finish();

        let mut immediates = self.immediates.iter().cloned();
        let mut params = self.params.iter().cloned().map(Id);

        program
            .instructions
            .extend(self.results.iter().zip(&library.components).map(|(n, c)| {
                let imm_arity = c.immediates();
                let immediates: Vec<_> = immediates.by_ref().take(imm_arity).collect();

                let arity = c.arity();
                let operands: Vec<_> = params.by_ref().take(arity).collect();

                let operator = c.make_operator(&immediates, &operands);
                let result = Id(*n);
                Instruction { result, operator }
            }));

        program.instructions.sort_unstable_by_key(|i| i.result.0);
        program
    }
}

#[derive(Debug)]
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

enum Verification {
    WorksForAllInputs,
    FailsOnInputs(Vec<i32>),
}

impl Program {
    pub fn synthesize<'a>(
        context: &'a z3::Context,
        spec: &Specification,
        library: &Library,
    ) -> Result<Program, Error> {
        if library.components.is_empty() {
            return Err(Error::NoComponents);
        }

        // Arbitrarily choose the initial inputs for finite synthesis.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut inputs: HashSet<Vec<i32>> = vec![(0..spec.arity).map(|_| rng.gen()).collect()]
            .into_iter()
            .collect();

        loop {
            println!(
                "############################################################################"
            );
            dbg!(&inputs);
            let assignments = Self::finite_synthesis(context, library, spec, &inputs)?;
            match Self::verification(context, library, spec, &assignments)? {
                Verification::WorksForAllInputs => {
                    return Ok(assignments.to_program(spec.arity, library))
                }
                Verification::FailsOnInputs(new_inputs) => {
                    let is_new = inputs.insert(dbg!(new_inputs));
                    assert!(is_new);
                    continue;
                }
            }
        }
    }

    fn finite_synthesis<'a>(
        context: &'a z3::Context,
        library: &Library,
        spec: &Specification,
        inputs: &HashSet<Vec<i32>>,
    ) -> Result<Assignments, Error> {
        let locations = library.fresh_locations(context, inputs.iter().next().unwrap().len());
        let immediates = library.fresh_immediates(context);
        let wfp = Self::well_formed_program(context, library, spec, &locations);

        let mut works_for_inputs = Vec::with_capacity(inputs.len() * 4);

        for input in inputs {
            let params = library.fresh_param_vars(context);
            let results = library.fresh_result_vars(context);
            let inputs: Vec<_> = input
                .iter()
                .map(|i| BitVec::from_i64(context, *i as i64, 32))
                .collect();
            let output = fresh_output(context);

            let lib = Self::library(context, library, &immediates, &params, &results);
            works_for_inputs.push(lib);

            let conn = Self::connectivity(context, &locations, &inputs, &output, &params, &results);
            works_for_inputs.push(conn);

            // TODO FITZGEN...
            // works_for_inputs.push(output._eq(&results.last().unwrap()));

            let spec = (spec.make_expression)(context, &inputs, &output);
            works_for_inputs.push(spec);
        }

        let works_for_inputs: Vec<&_> = works_for_inputs.iter().collect();

        let solver = z3::Solver::new(context);
        println!("finite synthesis =");
        solver.assert(&dbg!(wfp.and(&works_for_inputs)));

        match solver.check() {
            z3::SatResult::Unknown => Err(Error::SynthesisUnknown),
            z3::SatResult::Unsat => Err(Error::SynthesisUnsatisfiable),
            z3::SatResult::Sat => {
                let model = solver.get_model();
                println!("{}", model);

                let immediates = immediates
                    .into_iter()
                    .map(|imm| {
                        model
                            .eval(&imm)
                            .expect("should have a value for immediate")
                            .as_i64()
                            .expect("immediate should be convertible to i64")
                            as i32
                    })
                    .collect();
                let params = locations
                    .params
                    .iter()
                    .map(|p| model.eval(p).unwrap().as_u64().unwrap() as usize)
                    .collect();
                let results = locations
                    .results
                    .iter()
                    .map(|r| model.eval(r).unwrap().as_u64().unwrap() as usize)
                    .collect();
                let locations = Assignments {
                    immediates,
                    params,
                    results,
                };
                println!(
                    "{}",
                    locations.to_program(inputs.iter().next().unwrap().len(), library)
                );

                Ok(locations)
            }
        }
    }

    fn verification<'a>(
        context: &'a z3::Context,
        library: &Library,
        spec: &Specification,
        assignments: &Assignments,
    ) -> Result<Verification, Error> {
        let (locations, immediates) = assignments.to_vars(context, spec.arity);

        let inputs: Vec<_> = (0..spec.arity).map(|_| fresh_input(context)).collect();
        let output = fresh_output(context);
        let params = library.fresh_param_vars(context);
        let results = library.fresh_result_vars(context);

        let lib = Self::library(context, library, &immediates, &params, &results);
        let conn = Self::connectivity(context, &locations, &inputs, &output, &params, &results);

        let spec = (spec.make_expression)(context, &inputs, &output);
        let not_spec = spec.not();

        let solver = z3::Solver::new(context);
        println!("verification =");
        solver.assert(&dbg!(lib.and(&[&conn, &not_spec])));

        match solver.check() {
            z3::SatResult::Unknown => Err(Error::SynthesisUnknown),
            // There are no more inputs that don't satisfy the spec! We're done!
            z3::SatResult::Unsat => Ok(Verification::WorksForAllInputs),
            // There still exist inputs for which the synthesized program does
            // not fulfill the spec.
            z3::SatResult::Sat => {
                let model = solver.get_model();
                Ok(Verification::FailsOnInputs(
                    inputs
                        .iter()
                        .map(|i| model.eval(i).unwrap().as_i64().unwrap() as i32)
                        .collect(),
                ))
            }
        }
    }

    /// 5.1 Encoding Well-formed Programs
    fn well_formed_program<'a>(
        context: &'a z3::Context,
        library: &Library,
        spec: &Specification,
        locations: &LocationVars<'a>,
    ) -> Bool<'a> {
        let mut wfp = Vec::with_capacity(
            // Acyclic and consistent.
            2
                // Assignment of inputs.
                + locations.inputs.len()
                // Lower and upper bounds on params.
                + locations.params.len() * 2
                // Lower and upper bounds on results.
                + locations.results.len() * 2,
        );

        wfp.push(Self::consistent(context, locations));
        wfp.push(Self::acyclic(context, library, locations));

        let i_len = Int::from_i64(context, spec.arity as i64);
        let m = Int::from_i64(context, (locations.results.len() + spec.arity) as i64);
        let zero = Int::from_i64(context, 0);

        for (i, l) in locations.inputs.iter().enumerate() {
            let i = Int::from_i64(context, i as i64);
            wfp.push(l._eq(&i));
        }

        for l in &locations.params {
            // 0 <= l
            wfp.push(zero.le(l));
            // l < M
            wfp.push(l.lt(&m));
        }

        for l in &locations.results {
            // |i| <= l
            wfp.push(i_len.le(l));
            // l < m
            wfp.push(l.lt(&m));
        }

        let wfp: Vec<&_> = wfp.iter().collect();
        Bool::from_bool(context, true).and(&wfp)
    }

    fn consistent<'a>(context: &'a z3::Context, locations: &LocationVars<'a>) -> Bool<'a> {
        use itertools::Itertools;
        locations
            .results
            .iter()
            .tuple_combinations()
            .fold(Bool::from_bool(context, true), |cons, (x, y)| {
                cons.and(&[&x._eq(y).not()])
            })
    }

    fn acyclic<'a>(
        context: &'a z3::Context,
        library: &Library,
        locations: &LocationVars<'a>,
    ) -> Bool<'a> {
        let mut acycs = vec![];
        let mut params = locations.params.iter();
        let mut results = locations.results.iter();

        for c in &library.components {
            let result_location = results.next().unwrap();
            for _ in 0..c.arity() {
                let param_location = params.next().unwrap();
                acycs.push(param_location.lt(result_location));
            }
        }

        let acycs: Vec<&_> = acycs.iter().collect();
        Bool::from_bool(context, true).and(&acycs)
    }

    /// 5.2 Encoding Dataflow in Programs
    fn connectivity<'a>(
        context: &'a z3::Context,
        locations: &LocationVars<'a>,
        inputs: &[BitVec<'a>],
        output: &BitVec<'a>,
        params: &[BitVec<'a>],
        results: &[BitVec<'a>],
    ) -> Bool<'a> {
        let locs_to_vars: Vec<_> = locations
            .inputs
            .iter()
            .zip(inputs)
            .chain(locations.params.iter().zip(params))
            .chain(locations.results.iter().zip(results))
            .collect();

        let mut conn =
            Vec::with_capacity(locs_to_vars.len() * locs_to_vars.len() + locs_to_vars.len());

        let last_loc = Int::from_i64(context, inputs.len() as i64 + results.len() as i64 - 1);
        for (i, (l_x, x)) in locs_to_vars.iter().enumerate() {
            conn.push(l_x._eq(&last_loc).implies(&x._eq(output)));

            for (j, (l_y, y)) in locs_to_vars.iter().enumerate() {
                if i == j {
                    continue;
                }
                conn.push(l_x._eq(l_y).implies(&x._eq(y)));
            }
        }

        let conn: Vec<&_> = conn.iter().collect();
        Bool::from_bool(context, true).and(&conn)
    }

    fn library<'a>(
        context: &'a z3::Context,
        library: &Library,
        immediates: &[BitVec<'a>],
        params: &[BitVec<'a>],
        results: &[BitVec<'a>],
    ) -> Bool<'a> {
        let mut exprs = Vec::with_capacity(library.components.len());
        let mut immediates = immediates;
        let mut params = params;
        let mut results = results.iter();

        for c in &library.components {
            let (imms, rest) = immediates.split_at(c.immediates());
            immediates = rest;

            let (inputs, rest) = params.split_at(c.arity());
            params = rest;

            let result = results.next().unwrap();

            exprs.push(c.make_expression(context, imms, inputs, result));
        }

        let exprs: Vec<&_> = exprs.iter().collect();
        Bool::from_bool(context, true).and(&exprs)
    }

    pub fn dce(&mut self) {
        let mut used = HashSet::new();
        used.insert(self.instructions.last().unwrap().result);

        for inst in self.instructions.iter().rev() {
            if !used.contains(&inst.result) {
                continue;
            }

            inst.operator.operands(|op| {
                used.insert(op);
            });
        }

        self.instructions.retain(|inst| used.contains(&inst.result));

        let mut renumbering = HashMap::new();
        for (i, inst) in self.instructions.iter_mut().enumerate() {
            inst.operator.operands_mut(|x| *x = renumbering[x]);

            let old = renumbering.insert(inst.result, Id(i));
            debug_assert!(old.is_none());
            inst.result = Id(i);
        }
    }
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

        // let library = Library {
        //     components: vec![component::mul(), component::const_()],
        // };

        // let spec = Specification {
        //     arity: 1,
        //     make_expression: |_context, inputs, output| inputs[0].bvadd(&inputs[0])._eq(&output),
        // };

        // let p = Program::synthesize(&context, &spec, &library);

        let library = Library {
            components: vec![
                component::add(),
                component::mul(),
                component::const_(),
                component::add(),
            ],
        };

        let spec = Specification {
            arity: 1,
            make_expression: |context, inputs, output| {
                let three = BitVec::from_i64(context, 3, 32);
                inputs[0].bvmul(&three)._eq(&output)
            },
        };

        let p = Program::synthesize(&context, &spec, &library);
        dbg!(p);
    }
}
