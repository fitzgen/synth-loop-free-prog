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
use std::iter::FromIterator;
use std::ops::Range;
use z3::ast::{Ast, Bool, BV as BitVec};

fn fresh_immediate(context: &z3::Context, bit_width: u32) -> BitVec {
    BitVec::fresh_const(context, "immediate", bit_width)
}

fn fresh_param(context: &z3::Context, bit_width: u32) -> BitVec {
    BitVec::fresh_const(context, "param", bit_width)
}

fn fresh_result(context: &z3::Context, bit_width: u32) -> BitVec {
    BitVec::fresh_const(context, "result", bit_width)
}

fn fresh_input(context: &z3::Context, bit_width: u32) -> BitVec {
    BitVec::fresh_const(context, "input", bit_width)
}

fn fresh_output(context: &z3::Context, bit_width: u32) -> BitVec {
    BitVec::fresh_const(context, "output", bit_width)
}

#[derive(Debug)]
pub enum Error {
    NoComponents,
    LibraryTooLarge,
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

pub trait Specification {
    fn arity(&self) -> usize;

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        inputs: &[BitVec<'a>],
        output: &BitVec<'a>,
        bit_width: u32,
    ) -> Bool<'a>;
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
    /// Create a library of components that is roughly equivalent to the Brahma
    /// standard library.
    pub fn brahma_std() -> Self {
        Library {
            components: vec![
                component::and(),
                component::or(),
                component::xor(),
                component::shl(),
                component::shr_u(),
                component::eq(),
                component::ne(),
                component::add(),
                component::sub(),
                // add1
                component::const_(Some(1)),
                component::add(),
                // not(a) = a xor MAX
                component::const_(Some(std::u64::MAX)),
                component::xor(),
            ],
        }
    }
}

type Line<'a> = BitVec<'a>;

const MAX_LIBRARY_LEN: usize = std::u8::MAX as usize;
const LINE_BITS: u32 = 8;

fn line_from_u32<'a>(context: &'a z3::Context, line: u32) -> Line<'a> {
    BitVec::from_i64(context, line as i64, LINE_BITS)
}

fn fresh_line<'a>(context: &'a z3::Context, name: &str) -> Line<'a> {
    BitVec::fresh_const(context, name, LINE_BITS)
}

fn line_lt<'a>(lhs: &Line<'a>, rhs: &Line<'a>) -> Bool<'a> {
    lhs.bvult(rhs)
}

fn line_le<'a>(lhs: &Line<'a>, rhs: &Line<'a>) -> Bool<'a> {
    lhs.bvule(rhs)
}

#[derive(Debug)]
struct LocationVars<'a> {
    inputs: Vec<Line<'a>>,
    params: Vec<Line<'a>>,
    results: Vec<Line<'a>>,
}

impl<'a> LocationVars<'a> {
    fn new(context: &'a z3::Context, library: &Library, num_inputs: usize) -> Self {
        let inputs = (0..num_inputs)
            .map(|_| fresh_line(context, "input_location"))
            .collect();
        let params = library
            .components
            .iter()
            .flat_map(|c| (0..c.arity()).map(|_| fresh_line(context, "param_location")))
            .collect();
        let results = library
            .components
            .iter()
            .map(|_| fresh_line(context, "result_location"))
            .collect();
        LocationVars {
            inputs,
            params,
            results,
        }
    }

    fn inputs_range(&self) -> Range<usize> {
        0..self.inputs.len()
    }

    fn params_range(&self) -> Range<usize> {
        let start = self.inputs.len();
        let end = self.inputs.len() + self.params.len();
        start..end
    }

    fn results_range(&self) -> Range<usize> {
        let start = self.inputs.len() + self.params.len();
        let end = start + self.results.len();
        start..end
    }

    fn invalid_connections(&self, library: &Library) -> HashSet<(usize, usize)> {
        let mut invalid_connections = HashSet::new();

        // We never assign an input's location to another input's location, so
        // don't even consider these connections.
        for (i, a) in self.inputs_range().enumerate() {
            for b in self.inputs_range().skip(i) {
                invalid_connections.insert((a, b));
            }
        }

        // Similarly, a well-formed program will never assign a param's location
        // as another param; it should only be one of the original inputs or the
        // result of another component.
        for (i, p) in self.params_range().enumerate() {
            for q in self.params_range().skip(i) {
                invalid_connections.insert((p, q));
            }
        }

        // Finally, a well-formed will never have a component with its own
        // result as a parameter.
        let params = &mut self.params_range();
        for (r, c) in self.results_range().zip(&library.components) {
            for p in params.take(c.arity()) {
                invalid_connections.insert((r, p));
            }
        }

        invalid_connections
    }

    /// 5.1 Encoding Well-formed Programs
    fn well_formed_program(
        &self,
        context: &'a z3::Context,
        library: &Library,
        invalid_connections: &mut HashSet<(usize, usize)>,
    ) -> Bool<'a> {
        let mut wfp = Vec::with_capacity(
            // Acyclic and consistent.
            2
                // Assignment of inputs.
                + self.inputs.len()
                // Lower and upper bounds on params.
                + self.params.len() * 2
                // Lower and upper bounds on results.
                + self.results.len() * 2,
        );

        wfp.push(self.consistent(context, invalid_connections));
        wfp.push(self.acyclic(context, library));

        let i_len = line_from_u32(context, self.inputs.len() as u32);
        let m = line_from_u32(context, (self.results.len() + self.inputs.len()) as u32);
        let zero = line_from_u32(context, 0);

        for (i, l) in self.inputs.iter().enumerate() {
            let i = line_from_u32(context, i as u32);
            wfp.push(l._eq(&i));
        }

        for l in &self.params {
            // 0 <= l
            wfp.push(line_le(&zero, l));
            // l < M
            wfp.push(line_lt(l, &m));
        }

        for l in &self.results {
            // |i| <= l
            wfp.push(line_le(&i_len, l));
            // l < m
            wfp.push(line_lt(l, &m));
        }

        let wfp: Vec<&_> = wfp.iter().collect();
        Bool::from_bool(context, true).and(&wfp)
    }

    fn consistent(
        &self,
        context: &'a z3::Context,
        invalid_connections: &mut HashSet<(usize, usize)>,
    ) -> Bool<'a> {
        let mut cons = vec![];
        for (i, (i_x, x)) in self.results_range().zip(&self.results).enumerate() {
            for (i_y, y) in self.results_range().zip(&self.results).skip(i + 1) {
                invalid_connections.insert((i_x, i_y));
                cons.push(x._eq(y).not());
            }
        }
        let cons: Vec<&_> = cons.iter().collect();
        Bool::from_bool(context, true).and(&cons)
    }

    fn acyclic(&self, context: &'a z3::Context, library: &Library) -> Bool<'a> {
        let mut acycs = vec![];
        let mut params = self.params.iter();
        let mut results = self.results.iter();

        for c in &library.components {
            let result_location = results.next().unwrap();
            for _ in 0..c.arity() {
                let param_location = params.next().unwrap();
                acycs.push(line_lt(param_location, result_location));
            }
        }

        let acycs: Vec<&_> = acycs.iter().collect();
        Bool::from_bool(context, true).and(&acycs)
    }
}

#[derive(Debug)]
struct Assignments {
    immediates: Vec<u64>,
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
        bit_width: u32,
    ) -> (LocationVars<'a>, Vec<BitVec<'a>>) {
        let inputs = (0..num_inputs)
            .map(|i| line_from_u32(context, i as u32))
            .collect();
        let immediates = self
            .immediates
            .iter()
            .map(|imm| BitVec::from_i64(context, *imm as i64, bit_width))
            .collect();
        let params = self
            .params
            .iter()
            .map(|p| line_from_u32(context, *p as u32))
            .collect();
        let results = self
            .results
            .iter()
            .map(|r| line_from_u32(context, *r as u32))
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
    FailsOnInputs(Vec<u64>),
}

struct Synthesizer<'a, S>
where
    S: 'a + Specification,
{
    context: &'a z3::Context,
    library: &'a Library,
    spec: &'a S,
    locations: LocationVars<'a>,
    well_formed_program: Bool<'a>,
    invalid_connections: HashSet<(usize, usize)>,
}

impl<'a, S> Synthesizer<'a, S>
where
    S: 'a + Specification,
{
    fn new(context: &'a z3::Context, library: &'a Library, spec: &'a S) -> Result<Self, Error> {
        if library.components.is_empty() {
            return Err(Error::NoComponents);
        }
        if library.components.len() > MAX_LIBRARY_LEN {
            return Err(Error::LibraryTooLarge);
        }

        let locations = LocationVars::new(context, library, spec.arity());
        let mut invalid_connections = locations.invalid_connections(library);
        let well_formed_program =
            locations.well_formed_program(context, library, &mut invalid_connections);
        Ok(Synthesizer {
            context,
            library,
            spec,
            locations,
            well_formed_program,
            invalid_connections,
        })
    }

    fn is_invalid_connection(&self, i: usize, j: usize) -> bool {
        debug_assert!(
            i < self.locations.inputs.len()
                + self.locations.params.len()
                + self.locations.results.len()
        );
        debug_assert!(
            j < self.locations.inputs.len()
                + self.locations.params.len()
                + self.locations.results.len()
        );
        self.invalid_connections.contains(&(i, j)) || self.invalid_connections.contains(&(j, i))
    }

    fn fresh_immediates(&self, bit_width: u32) -> Vec<BitVec<'a>> {
        self.library
            .components
            .iter()
            .flat_map(|c| (0..c.immediates()).map(|_| fresh_immediate(self.context, bit_width)))
            .collect()
    }

    fn fresh_param_vars(&self, bit_width: u32) -> Vec<BitVec<'a>> {
        self.library
            .components
            .iter()
            .flat_map(|c| (0..c.arity()).map(|_| fresh_param(self.context, bit_width)))
            .collect()
    }

    fn fresh_result_vars(&self, bit_width: u32) -> Vec<BitVec<'a>> {
        self.library
            .components
            .iter()
            .map(|_| fresh_result(self.context, bit_width))
            .collect()
    }

    fn finite_synthesis(
        &self,
        inputs: &HashSet<Vec<u64>>,
        bit_width: u32,
    ) -> Result<Assignments, Error> {
        let immediates = self.fresh_immediates(bit_width);
        let mut works_for_inputs = Vec::with_capacity(inputs.len() * 4);

        for input in inputs {
            let params = self.fresh_param_vars(bit_width);
            let results = self.fresh_result_vars(bit_width);
            let inputs: Vec<_> = input
                .iter()
                .map(|i| BitVec::from_i64(self.context, *i as i64, bit_width))
                .collect();
            let output = fresh_output(self.context, bit_width);

            let lib = self.library(&immediates, &params, &results, bit_width);
            works_for_inputs.push(lib);

            let conn = self.connectivity(&inputs, &output, &params, &results);
            works_for_inputs.push(conn);

            let spec = self
                .spec
                .make_expression(self.context, &inputs, &output, bit_width);
            works_for_inputs.push(spec);
        }

        let works_for_inputs: Vec<&_> = works_for_inputs.iter().collect();

        let solver = z3::Solver::new(self.context);
        solver.assert(&self.well_formed_program.and(&works_for_inputs));

        match solver.check() {
            z3::SatResult::Unknown => Err(Error::SynthesisUnknown),
            z3::SatResult::Unsat => Err(Error::SynthesisUnsatisfiable),
            z3::SatResult::Sat => {
                let model = solver.get_model();

                let immediates = immediates
                    .into_iter()
                    .map(|imm| {
                        model
                            .eval(&imm)
                            .expect("should have a value for immediate")
                            .as_i64()
                            .expect("immediate should be convertible to i64")
                            as u64
                    })
                    .collect();
                let params = self
                    .locations
                    .params
                    .iter()
                    .map(|p| model.eval(p).unwrap().as_u64().unwrap() as usize)
                    .collect();
                let results = self
                    .locations
                    .results
                    .iter()
                    .map(|r| model.eval(r).unwrap().as_u64().unwrap() as usize)
                    .collect();
                Ok(Assignments {
                    immediates,
                    params,
                    results,
                })
            }
        }
    }

    fn verification(
        &self,
        assignments: &Assignments,
        bit_width: u32,
    ) -> Result<Verification, Error> {
        let inputs: Vec<_> = (0..self.spec.arity())
            .map(|_| fresh_input(self.context, bit_width))
            .collect();
        let output = fresh_output(self.context, bit_width);

        let mut prog = assignments.to_program(self.spec.arity(), self.library);
        prog.dce();
        let prog = prog.make_expression(self.context, &inputs, &output, bit_width);

        let spec = self
            .spec
            .make_expression(self.context, &inputs, &output, bit_width);
        let not_spec = spec.not();

        let solver = z3::Solver::new(self.context);
        solver.assert(&prog.and(&[&not_spec]));

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
                        .map(|i| model.eval(i).unwrap().as_i64().unwrap() as u64)
                        .collect(),
                ))
            }
        }
    }

    /// 5.2 Encoding Dataflow in Programs
    fn connectivity(
        &self,
        inputs: &[BitVec<'a>],
        output: &BitVec<'a>,
        params: &[BitVec<'a>],
        results: &[BitVec<'a>],
    ) -> Bool<'a> {
        let locs_to_vars: Vec<_> = self
            .locations
            .inputs
            .iter()
            .zip(inputs)
            .chain(self.locations.params.iter().zip(params))
            .chain(self.locations.results.iter().zip(results))
            .collect();

        let mut conn =
            Vec::with_capacity(locs_to_vars.len() * locs_to_vars.len() + locs_to_vars.len());

        let last_loc = line_from_u32(self.context, inputs.len() as u32 + results.len() as u32 - 1);
        for (i, (l_x, x)) in locs_to_vars.iter().enumerate() {
            conn.push(l_x._eq(&last_loc).implies(&x._eq(output)));

            for (j, (l_y, y)) in locs_to_vars.iter().enumerate().skip(i + 1) {
                if self.is_invalid_connection(i, j) {
                    continue;
                }
                conn.push(l_x._eq(l_y).implies(&x._eq(y)));
            }
        }

        let conn: Vec<&_> = conn.iter().collect();
        Bool::from_bool(self.context, true).and(&conn)
    }

    fn library(
        &self,
        immediates: &[BitVec<'a>],
        params: &[BitVec<'a>],
        results: &[BitVec<'a>],
        bit_width: u32,
    ) -> Bool<'a> {
        let mut exprs = Vec::with_capacity(self.library.components.len());
        let mut immediates = immediates;
        let mut params = params;
        let mut results = results.iter();

        for c in &self.library.components {
            let (imms, rest) = immediates.split_at(c.immediates());
            immediates = rest;

            let (inputs, rest) = params.split_at(c.arity());
            params = rest;

            let result = results.next().unwrap();

            exprs.push(
                c.make_expression(self.context, imms, inputs, bit_width)
                    ._eq(result),
            );
        }

        let exprs: Vec<&_> = exprs.iter().collect();
        Bool::from_bool(self.context, true).and(&exprs)
    }

    fn synthesize(&self) -> Result<Program, Error> {
        // Arbitrarily choose the initial inputs for finite synthesis.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut inputs: HashSet<Vec<u64>> = HashSet::new();
        inputs.insert((0..self.spec.arity()).map(|_| rng.gen::<u64>()).collect());

        let mut bit_width = 2;
        'cegis: loop {
            // dbg!(&inputs);
            let assignments = self.finite_synthesis(&inputs, bit_width)?;

            // let mut p = assignments.to_program(self.spec.arity(), self.library);
            // p.dce();
            // println!("-------------\n{}-----------------", p);

            let mut verifying_with_more_bits = false;
            loop {
                // println!("verifying at bit width = {}", bit_width);
                match self.verification(&assignments, bit_width)? {
                    Verification::WorksForAllInputs => {
                        debug_assert!(bit_width <= 32);
                        debug_assert!(bit_width.is_power_of_two());
                        if bit_width == 32 {
                            return Ok(assignments.to_program(self.spec.arity(), self.library));
                        } else {
                            // println!("verified at bit width = {}", bit_width);
                            bit_width *= 2;
                            verifying_with_more_bits = true;
                            // TODO: if the synthesized assignments use
                            // immediate constants, try to extend the constants
                            // for the wider bit width in various ways. See
                            // *Program Synthesis for Program Analysis*.
                        }
                    }
                    Verification::FailsOnInputs(new_inputs) => {
                        // println!("counter example found at bit width = {}", bit_width);
                        let is_new = inputs.insert(new_inputs);
                        assert!(is_new || verifying_with_more_bits);
                        continue 'cegis;
                    }
                }
            }
        }
    }
}

impl Program {
    pub fn synthesize<'a>(
        context: &'a z3::Context,
        spec: &impl Specification,
        library: &Library,
    ) -> Result<Program, Error> {
        let synthesizer = Synthesizer::new(context, library, spec)?;
        synthesizer.synthesize()
    }

    pub fn dce(&mut self) {
        let mut used: HashSet<Id> = HashSet::from_iter(
            self.instructions
                .iter()
                .take_while(|inst| inst.operator == Operator::Var)
                .map(|inst| inst.result)
                .chain(Some(self.instructions.last().unwrap().result)),
        );

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

impl Specification for Program {
    fn arity(&self) -> usize {
        self.instructions
            .iter()
            .take_while(|inst| inst.operator == Operator::Var)
            .count()
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        inputs: &[BitVec<'a>],
        output: &BitVec<'a>,
        bit_width: u32,
    ) -> Bool<'a> {
        assert!(self.instructions.len() > inputs.len());

        let mut vars: Vec<_> = inputs.iter().cloned().collect();

        let mut immediates = vec![];
        let mut operands = vec![];
        for instr in self.instructions.iter().skip(inputs.len()) {
            immediates.clear();
            instr.operator.immediates(|imm| {
                immediates.push(BitVec::from_i64(context, imm as i64, bit_width))
            });

            operands.clear();
            instr
                .operator
                .operands(|Id(x)| operands.push(vars[x].clone()));

            vars.push(
                instr
                    .operator
                    .make_expression(context, &immediates, &operands, bit_width),
            );
        }

        vars.pop().unwrap()._eq(output)
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
             %2 = const 0x1\n\
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

        let library = Library::brahma_std();

        // let library = Library {
        //     components: vec![
        //         component::const_(Some(1)),
        //         component::sub(),
        //         component::and(),
        //     ],
        // };

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

        // let a = builder.var();
        // let b = builder.const_(1);
        // let c = builder.sub(a, b);
        // let _ = builder.and(a, c);

        let a = builder.var();
        let _ = builder.popcnt(a);

        let spec = builder.finish();

        let mut p = Program::synthesize(&context, &spec, &library).unwrap();
        // println!("Synthesized:\n\n{}", p);
        p.dce();
        // println!("DCE'd:\n\n{}", p);
    }
}
