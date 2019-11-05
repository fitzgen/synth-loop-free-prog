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
    fn arity(&self) -> usize;

    fn make_operator(&self, immediates: &[u64], operands: &[Id]) -> Operator;

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        bit_width: u32,
    ) -> BitVec<'a>;

    /// How many immediates does this component require?
    fn immediates(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct Const(Option<u64>);

impl Component for Const {
    fn arity(&self) -> usize {
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

    fn immediates(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
            zero_bit: &BitVec<'a>,
            bit_width: u32,
            i: u32,
        ) -> BitVec<'a> {
            if i == bit_width {
                bit_vec_from_u64(context, i as u64, bit_width)
            } else {
                input.extract(i, i)._eq(&zero_bit).ite(
                    &bit_vec_from_u64(context, i as u64, bit_width),
                    &clz(context, input, zero_bit, bit_width, i + 1),
                )
            }
        }

        let zero_bit = BitVec::from_i64(context, 0, 1);
        clz(context, &operands[0], &zero_bit, bit_width, 0)
    }
}

pub fn clz() -> Box<dyn Component> {
    Box::new(Clz) as _
}

#[derive(Debug)]
struct Ctz;

impl Component for Ctz {
    fn arity(&self) -> usize {
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
            zero_bit: &BitVec<'a>,
            bit_width: u32,
            i: u32,
        ) -> BitVec<'a> {
            if i == bit_width {
                bit_vec_from_u64(context, i as u64, bit_width)
            } else {
                input
                    .extract(bit_width - i - 1, bit_width - i - 1)
                    ._eq(&zero_bit)
                    .ite(
                        &bit_vec_from_u64(context, i as u64, bit_width),
                        &ctz(context, input, zero_bit, bit_width, i + 1),
                    )
            }
        }

        let zero_bit = BitVec::from_i64(context, 0, 1);
        ctz(context, &operands[0], &zero_bit, bit_width, 0)
    }
}

pub fn ctz() -> Box<dyn Component> {
    Box::new(Ctz) as _
}

#[derive(Debug)]
struct Popcnt;

impl Component for Popcnt {
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
    fn arity(&self) -> usize {
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
        }
    };
}

impl Component for Operator {
    fn arity(&self) -> usize {
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

    fn immediates(&self) -> usize {
        with_operator_component!(self, |c| c.immediates())
    }
}
