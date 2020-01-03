use crate::Id;
use std::fmt::{self, Display};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operator {
    // Declare a new variable.
    Var,

    // A constant value.
    Const(u64),

    // Unary operators.
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

    // If-then-else.
    Select(Id, Id, Id),
}

impl Operator {
    pub fn arity(&self) -> usize {
        match self {
            Operator::Var | Operator::Const(_) => 0,
            Operator::Eqz(_) | Operator::Clz(_) | Operator::Ctz(_) | Operator::Popcnt(_) => 1,
            Operator::Eq(_, _)
            | Operator::Ne(_, _)
            | Operator::LtS(_, _)
            | Operator::LtU(_, _)
            | Operator::GtS(_, _)
            | Operator::GtU(_, _)
            | Operator::LeS(_, _)
            | Operator::LeU(_, _)
            | Operator::GeS(_, _)
            | Operator::GeU(_, _)
            | Operator::Add(_, _)
            | Operator::Sub(_, _)
            | Operator::Mul(_, _)
            | Operator::DivS(_, _)
            | Operator::DivU(_, _)
            | Operator::RemS(_, _)
            | Operator::RemU(_, _)
            | Operator::And(_, _)
            | Operator::Or(_, _)
            | Operator::Xor(_, _)
            | Operator::Shl(_, _)
            | Operator::ShrS(_, _)
            | Operator::ShrU(_, _)
            | Operator::Rotl(_, _)
            | Operator::Rotr(_, _) => 2,
            Operator::Select(_, _, _) => 3,
        }
    }

    pub fn immediates(&self, mut f: impl FnMut(u64)) {
        if let Operator::Const(c) = *self {
            f(c);
        }
    }

    pub fn operands(&self, mut f: impl FnMut(Id)) {
        match *self {
            Operator::Var | Operator::Const(_) => {}
            Operator::Eqz(a) | Operator::Clz(a) | Operator::Ctz(a) | Operator::Popcnt(a) => f(a),
            Operator::Eq(a, b)
            | Operator::Ne(a, b)
            | Operator::LtS(a, b)
            | Operator::LtU(a, b)
            | Operator::GtS(a, b)
            | Operator::GtU(a, b)
            | Operator::LeS(a, b)
            | Operator::LeU(a, b)
            | Operator::GeS(a, b)
            | Operator::GeU(a, b)
            | Operator::Add(a, b)
            | Operator::Sub(a, b)
            | Operator::Mul(a, b)
            | Operator::DivS(a, b)
            | Operator::DivU(a, b)
            | Operator::RemS(a, b)
            | Operator::RemU(a, b)
            | Operator::And(a, b)
            | Operator::Or(a, b)
            | Operator::Xor(a, b)
            | Operator::Shl(a, b)
            | Operator::ShrS(a, b)
            | Operator::ShrU(a, b)
            | Operator::Rotl(a, b)
            | Operator::Rotr(a, b) => {
                f(a);
                f(b);
            }
            Operator::Select(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
        }
    }

    pub fn operands_mut(&mut self, mut f: impl FnMut(&mut Id)) {
        match self {
            Operator::Var | Operator::Const(_) => {}
            Operator::Eqz(a) | Operator::Clz(a) | Operator::Ctz(a) | Operator::Popcnt(a) => f(a),
            Operator::Eq(a, b)
            | Operator::Ne(a, b)
            | Operator::LtS(a, b)
            | Operator::LtU(a, b)
            | Operator::GtS(a, b)
            | Operator::GtU(a, b)
            | Operator::LeS(a, b)
            | Operator::LeU(a, b)
            | Operator::GeS(a, b)
            | Operator::GeU(a, b)
            | Operator::Add(a, b)
            | Operator::Sub(a, b)
            | Operator::Mul(a, b)
            | Operator::DivS(a, b)
            | Operator::DivU(a, b)
            | Operator::RemS(a, b)
            | Operator::RemU(a, b)
            | Operator::And(a, b)
            | Operator::Or(a, b)
            | Operator::Xor(a, b)
            | Operator::Shl(a, b)
            | Operator::ShrS(a, b)
            | Operator::ShrU(a, b)
            | Operator::Rotl(a, b)
            | Operator::Rotr(a, b) => {
                f(a);
                f(b);
            }
            Operator::Select(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operator::Var => write!(f, "var"),
            Operator::Const(c) => write!(f, "const {:#X}", c),
            Operator::Eqz(id) => write!(f, "eqz {}", id),
            Operator::Clz(id) => write!(f, "clz {}", id),
            Operator::Ctz(id) => write!(f, "ctz {}", id),
            Operator::Popcnt(id) => write!(f, "popcnt {}", id),
            Operator::Eq(a, b) => write!(f, "eq {}, {}", a, b),
            Operator::Ne(a, b) => write!(f, "ne {}, {}", a, b),
            Operator::LtS(a, b) => write!(f, "lt_s {}, {}", a, b),
            Operator::LtU(a, b) => write!(f, "lt_u {}, {}", a, b),
            Operator::GtS(a, b) => write!(f, "gt_s {}, {}", a, b),
            Operator::GtU(a, b) => write!(f, "gt_u {}, {}", a, b),
            Operator::LeS(a, b) => write!(f, "le_s {}, {}", a, b),
            Operator::LeU(a, b) => write!(f, "le_u {}, {}", a, b),
            Operator::GeS(a, b) => write!(f, "ge_s {}, {}", a, b),
            Operator::GeU(a, b) => write!(f, "ge_u {}, {}", a, b),
            Operator::Add(a, b) => write!(f, "add {}, {}", a, b),
            Operator::Sub(a, b) => write!(f, "sub {}, {}", a, b),
            Operator::Mul(a, b) => write!(f, "mul {}, {}", a, b),
            Operator::DivS(a, b) => write!(f, "div_s {}, {}", a, b),
            Operator::DivU(a, b) => write!(f, "div_u {}, {}", a, b),
            Operator::RemS(a, b) => write!(f, "rem_s {}, {}", a, b),
            Operator::RemU(a, b) => write!(f, "rem_u {}, {}", a, b),
            Operator::And(a, b) => write!(f, "and {}, {}", a, b),
            Operator::Or(a, b) => write!(f, "or {}, {}", a, b),
            Operator::Xor(a, b) => write!(f, "xor {}, {}", a, b),
            Operator::Shl(a, b) => write!(f, "shl {}, {}", a, b),
            Operator::ShrS(a, b) => write!(f, "shr_s {}, {}", a, b),
            Operator::ShrU(a, b) => write!(f, "shr_u {}, {}", a, b),
            Operator::Rotl(a, b) => write!(f, "rotl {}, {}", a, b),
            Operator::Rotr(a, b) => write!(f, "rotr {}, {}", a, b),
            Operator::Select(a, b, c) => write!(f, "select {}, {}, {}", a, b, c),
        }
    }
}
