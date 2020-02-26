use crate::{Id, Operator};
use std::fmt::Debug;
use z3::ast::{Ast, BV as BitVec};

fn bit_vec_from_u64(context: &z3::Context, val: u64, bit_width: u32) -> BitVec {
    BitVec::from_i64(context, val as i64, bit_width)
}

fn zero(context: &z3::Context, bit_width: u32) -> BitVec {
    bit_vec_from_u64(context, 0, bit_width)
}

fn one(context: &z3::Context, bit_width: u32) -> BitVec {
    bit_vec_from_u64(context, 1, bit_width)
}

pub trait Component: Debug {
    fn operand_arity(&self) -> usize;

    fn make_operator(&self, immediates: &[u64], operands: &[Id]) -> Operator;

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a>;

    /// How many immediates does this component require?
    fn immediate_arity(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct Const(Option<u64>);

impl Component for Const {
    fn operand_arity(&self) -> usize {
        0
    }

    fn make_operator(&self, immediates: &[u64], _operands: &[Id]) -> Operator {
        if let Some(val) = self.0 {
            Operator::Const(val)
        } else {
            Operator::Const(immediates[0])
        }
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        immediates: &[BitVec<'a>],
        _operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        if let Some(val) = self.0 {
            BitVec::from_i64(context, val as i64, bit_width)
        } else {
            immediates[0].clone()
        }
    }

    fn immediate_arity(&self) -> usize {
        if self.0.is_some() {
            0
        } else {
            1
        }
    }
}

pub fn const_(val: Option<u64>) -> Box<dyn Component> {
    Box::new(Const(val)) as _
}

#[derive(Debug)]
struct Eqz;

impl Component for Eqz {
    fn operand_arity(&self) -> usize {
        1
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Eqz(operands[0])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        zero(context, bit_width)
            ._eq(&operands[0])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn eqz() -> Box<dyn Component> {
    Box::new(Eqz) as _
}

#[derive(Debug)]
struct Clz;

impl Component for Clz {
    fn operand_arity(&self) -> usize {
        1
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Clz(operands[0])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        fn clz<'a>(
            context: &'a z3::Context,
            input: &BitVec<'a>,
            one_bit: &BitVec<'a>,
            bit_width: u32,
            i: u32,
        ) -> BitVec<'a> {
            if i == bit_width {
                bit_vec_from_u64(context, i as u64, bit_width)
            } else {
                input
                    .extract(bit_width - 1 - i, bit_width - 1 - i)
                    ._eq(&one_bit)
                    .ite(
                        &bit_vec_from_u64(context, i as u64, bit_width),
                        &clz(context, input, one_bit, bit_width, i + 1),
                    )
            }
        }

        let one_bit = BitVec::from_i64(context, 1, 1);
        clz(context, &operands[0], &one_bit, bit_width, 0)
    }
}

pub fn clz() -> Box<dyn Component> {
    Box::new(Clz) as _
}

#[derive(Debug)]
struct Ctz;

impl Component for Ctz {
    fn operand_arity(&self) -> usize {
        1
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Ctz(operands[0])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        fn ctz<'a>(
            context: &'a z3::Context,
            input: &BitVec<'a>,
            one_bit: &BitVec<'a>,
            bit_width: u32,
            i: u32,
        ) -> BitVec<'a> {
            if i == bit_width {
                bit_vec_from_u64(context, i as u64, bit_width)
            } else {
                input.extract(i, i)._eq(&one_bit).ite(
                    &bit_vec_from_u64(context, i as u64, bit_width),
                    &ctz(context, input, one_bit, bit_width, i + 1),
                )
            }
        }

        let one_bit = BitVec::from_i64(context, 1, 1);
        ctz(context, &operands[0], &one_bit, bit_width, 0)
    }
}

pub fn ctz() -> Box<dyn Component> {
    Box::new(Ctz) as _
}

#[derive(Debug)]
struct Popcnt;

impl Component for Popcnt {
    fn operand_arity(&self) -> usize {
        1
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Popcnt(operands[0])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        let mut bits: Vec<_> = (0..bit_width)
            .map(|i| operands[0].extract(i, i).zero_ext(bit_width - 1))
            .collect();
        let initial = bits.pop().unwrap();
        bits.iter().fold(initial, |a, b| a.bvadd(b))
    }
}

pub fn popcnt() -> Box<dyn Component> {
    Box::new(Popcnt) as _
}

#[derive(Debug)]
struct Eq;

impl Component for Eq {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Eq(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            ._eq(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn eq() -> Box<dyn Component> {
    Box::new(Eq) as _
}

#[derive(Debug)]
struct Ne;

impl Component for Ne {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Ne(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            ._eq(&operands[1])
            .ite(&zero(context, bit_width), &one(context, bit_width))
    }
}

pub fn ne() -> Box<dyn Component> {
    Box::new(Ne) as _
}

#[derive(Debug)]
struct LtS;

impl Component for LtS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::LtS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvslt(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn lt_s() -> Box<dyn Component> {
    Box::new(LtS) as _
}

#[derive(Debug)]
struct LtU;

impl Component for LtU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::LtU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvult(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn lt_u() -> Box<dyn Component> {
    Box::new(LtU) as _
}

#[derive(Debug)]
struct GtS;

impl Component for GtS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::GtS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvsgt(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn gt_s() -> Box<dyn Component> {
    Box::new(GtS) as _
}

#[derive(Debug)]
struct GtU;

impl Component for GtU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::GtU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvugt(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn gt_u() -> Box<dyn Component> {
    Box::new(GtU) as _
}

#[derive(Debug)]
struct LeS;

impl Component for LeS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::LeS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvsle(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn le_s() -> Box<dyn Component> {
    Box::new(LeS) as _
}

#[derive(Debug)]
struct LeU;

impl Component for LeU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::LeU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvule(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn le_u() -> Box<dyn Component> {
    Box::new(LeU) as _
}

#[derive(Debug)]
struct GeS;

impl Component for GeS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::GeS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvsge(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn ge_s() -> Box<dyn Component> {
    Box::new(GeS) as _
}

#[derive(Debug)]
struct GeU;

impl Component for GeU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::GeU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            .bvuge(&operands[1])
            .ite(&one(context, bit_width), &zero(context, bit_width))
    }
}

pub fn ge_u() -> Box<dyn Component> {
    Box::new(GeU) as _
}

#[derive(Debug)]
struct Add;

impl Component for Add {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Add(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvadd(&operands[1])
    }
}

pub fn add() -> Box<dyn Component> {
    Box::new(Add) as _
}

#[derive(Debug)]
struct Sub;

impl Component for Sub {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Sub(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvsub(&operands[1])
    }
}

pub fn sub() -> Box<dyn Component> {
    Box::new(Sub) as _
}

#[derive(Debug)]
struct Mul;

impl Component for Mul {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Mul(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvmul(&operands[1])
    }
}

pub fn mul() -> Box<dyn Component> {
    Box::new(Mul) as _
}

#[derive(Debug)]
struct DivS;

impl Component for DivS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::DivS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvsdiv(&operands[1])
    }
}

pub fn div_s() -> Box<dyn Component> {
    Box::new(DivS) as _
}

#[derive(Debug)]
struct DivU;

impl Component for DivU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::DivU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvudiv(&operands[1])
    }
}

pub fn div_u() -> Box<dyn Component> {
    Box::new(DivU) as _
}

#[derive(Debug)]
struct RemS;

impl Component for RemS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::RemS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvsrem(&operands[1])
    }
}

pub fn rem_s() -> Box<dyn Component> {
    Box::new(RemS) as _
}

#[derive(Debug)]
struct RemU;

impl Component for RemU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::RemU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvurem(&operands[1])
    }
}

pub fn rem_u() -> Box<dyn Component> {
    Box::new(RemU) as _
}

#[derive(Debug)]
struct And;

impl Component for And {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::And(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvand(&operands[1])
    }
}

pub fn and() -> Box<dyn Component> {
    Box::new(And) as _
}

#[derive(Debug)]
struct Or;

impl Component for Or {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Or(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvor(&operands[1])
    }
}

pub fn or() -> Box<dyn Component> {
    Box::new(Or) as _
}

#[derive(Debug)]
struct Xor;

impl Component for Xor {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Xor(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvxor(&operands[1])
    }
}

pub fn xor() -> Box<dyn Component> {
    Box::new(Xor) as _
}

#[derive(Debug)]
struct Shl;

impl Component for Shl {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Shl(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvshl(&operands[1])
    }
}

pub fn shl() -> Box<dyn Component> {
    Box::new(Shl) as _
}

#[derive(Debug)]
struct ShrS;

impl Component for ShrS {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::ShrS(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvashr(&operands[1])
    }
}

pub fn shr_s() -> Box<dyn Component> {
    Box::new(ShrS) as _
}

#[derive(Debug)]
struct ShrU;

impl Component for ShrU {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::ShrU(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvlshr(&operands[1])
    }
}

pub fn shr_u() -> Box<dyn Component> {
    Box::new(ShrU) as _
}

#[derive(Debug)]
struct Rotl;

impl Component for Rotl {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Rotl(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvrotl(&operands[1])
    }
}

pub fn rotl() -> Box<dyn Component> {
    Box::new(Rotl) as _
}

#[derive(Debug)]
struct Rotr;

impl Component for Rotr {
    fn operand_arity(&self) -> usize {
        2
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Rotr(operands[0], operands[1])
    }

    fn make_expression<'a>(
        &self,
        _context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        _bit_width: u32,
    ) -> BitVec<'a> {
        operands[0].bvrotr(&operands[1])
    }
}

pub fn rotr() -> Box<dyn Component> {
    Box::new(Rotr) as _
}

#[derive(Debug)]
struct Select;

impl Component for Select {
    fn operand_arity(&self) -> usize {
        3
    }

    fn make_operator(&self, _immediates: &[u64], operands: &[Id]) -> Operator {
        Operator::Select(operands[0], operands[1], operands[2])
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        _immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        operands[0]
            ._eq(&zero(context, bit_width))
            .ite(&operands[2], &operands[1])
    }
}

pub fn select() -> Box<dyn Component> {
    Box::new(Select) as _
}

macro_rules! with_operator_component {
    ( $me:expr , |$c:ident| $body:expr ) => {
        match $me {
            Operator::Var => panic!("`Var` operators do not have a component"),
            Operator::Const(c) => {
                let $c = Const(Some(*c));
                $body
            }
            Operator::Eqz(_) => {
                let $c = Eqz;
                $body
            }
            Operator::Clz(_) => {
                let $c = Clz;
                $body
            }
            Operator::Ctz(_) => {
                let $c = Ctz;
                $body
            }
            Operator::Popcnt(_) => {
                let $c = Popcnt;
                $body
            }
            Operator::Eq(_, _) => {
                let $c = Eq;
                $body
            }
            Operator::Ne(_, _) => {
                let $c = Ne;
                $body
            }
            Operator::LtS(_, _) => {
                let $c = LtS;
                $body
            }
            Operator::LtU(_, _) => {
                let $c = LtU;
                $body
            }
            Operator::GtS(_, _) => {
                let $c = GtS;
                $body
            }
            Operator::GtU(_, _) => {
                let $c = GtU;
                $body
            }
            Operator::LeS(_, _) => {
                let $c = LeS;
                $body
            }
            Operator::LeU(_, _) => {
                let $c = LeU;
                $body
            }
            Operator::GeS(_, _) => {
                let $c = GeS;
                $body
            }
            Operator::GeU(_, _) => {
                let $c = GeU;
                $body
            }
            Operator::Add(_, _) => {
                let $c = Add;
                $body
            }
            Operator::Sub(_, _) => {
                let $c = Sub;
                $body
            }
            Operator::Mul(_, _) => {
                let $c = Mul;
                $body
            }
            Operator::DivS(_, _) => {
                let $c = DivS;
                $body
            }
            Operator::DivU(_, _) => {
                let $c = DivU;
                $body
            }
            Operator::RemS(_, _) => {
                let $c = RemS;
                $body
            }
            Operator::RemU(_, _) => {
                let $c = RemU;
                $body
            }
            Operator::And(_, _) => {
                let $c = And;
                $body
            }
            Operator::Or(_, _) => {
                let $c = Or;
                $body
            }
            Operator::Xor(_, _) => {
                let $c = Xor;
                $body
            }
            Operator::Shl(_, _) => {
                let $c = Shl;
                $body
            }
            Operator::ShrS(_, _) => {
                let $c = ShrS;
                $body
            }
            Operator::ShrU(_, _) => {
                let $c = ShrU;
                $body
            }
            Operator::Rotl(_, _) => {
                let $c = Rotl;
                $body
            }
            Operator::Rotr(_, _) => {
                let $c = Rotr;
                $body
            }
            Operator::Select(_, _, _) => {
                let $c = Select;
                $body
            }
        }
    };
}

impl Component for Operator {
    fn operand_arity(&self) -> usize {
        Operator::arity(self)
    }

    fn make_operator(&self, immediates: &[u64], operands: &[Id]) -> Operator {
        with_operator_component!(self, |c| c.make_operator(immediates, operands))
    }

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a> {
        with_operator_component!(self, |c| {
            c.make_expression(context, immediates, operands, bit_width)
        })
    }

    fn immediate_arity(&self) -> usize {
        with_operator_component!(self, |c| c.immediate_arity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctz_test() {
        let _ = env_logger::try_init();
        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);

        // 0000 0000 0000 0000 0000 0000 0000 0010
        assert!(ctz()
            .make_expression(&ctx, &vec![], &vec![bit_vec_from_u64(&ctx, 2, 32)], 32)
            ._eq(&bit_vec_from_u64(&ctx, 1, 32))
            .simplify()
            .as_bool()
            .unwrap());
        // all zeroes
        assert!(ctz()
            .make_expression(&ctx, &vec![], &vec![bit_vec_from_u64(&ctx, 0, 32)], 32)
            ._eq(&bit_vec_from_u64(&ctx, 32, 32))
            .simplify()
            .as_bool()
            .unwrap());
        // all ones
        assert!(ctz()
            .make_expression(
                &ctx,
                &vec![],
                &vec![z3::ast::BV::from_i64(&ctx, -1, 32)],
                32
            )
            ._eq(&bit_vec_from_u64(&ctx, 0, 32))
            .simplify()
            .as_bool()
            .unwrap());
        // 00 1010
        assert!(ctz()
            .make_expression(&ctx, &vec![], &vec![bit_vec_from_u64(&ctx, 10, 6)], 6)
            ._eq(&bit_vec_from_u64(&ctx, 1, 6))
            .simplify()
            .as_bool()
            .unwrap());
    }

    #[test]
    fn clz_test() {
        let _ = env_logger::try_init();
        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);

        // 0000 0000 0000 0000 0000 0000 0000 0010
        assert!(clz()
            .make_expression(&ctx, &vec![], &vec![bit_vec_from_u64(&ctx, 2, 32)], 32)
            ._eq(&bit_vec_from_u64(&ctx, 30, 32))
            .simplify()
            .as_bool()
            .unwrap());
        // all zeroes
        assert!(clz()
            .make_expression(&ctx, &vec![], &vec![bit_vec_from_u64(&ctx, 0, 32)], 32)
            ._eq(&bit_vec_from_u64(&ctx, 32, 32))
            .simplify()
            .as_bool()
            .unwrap());
        // all ones
        assert!(clz()
            .make_expression(
                &ctx,
                &vec![],
                &vec![z3::ast::BV::from_i64(&ctx, -1, 32)],
                32
            )
            ._eq(&bit_vec_from_u64(&ctx, 0, 32))
            .simplify()
            .as_bool()
            .unwrap());
        // 00 1010
        assert!(clz()
            .make_expression(&ctx, &vec![], &vec![bit_vec_from_u64(&ctx, 10, 6)], 6)
            ._eq(&bit_vec_from_u64(&ctx, 2, 6))
            .simplify()
            .as_bool()
            .unwrap());
    }
}
