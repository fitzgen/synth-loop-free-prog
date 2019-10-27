use crate::{Id, Operator};
use std::fmt::Debug;
use z3::ast::{Ast, Bool, BV as BitVec};

fn bit_vec_from_i32(context: &z3::Context, val: i32) -> BitVec {
    BitVec::from_i64(context, val as i64, 32)
}

fn zero(context: &z3::Context) -> BitVec {
    bit_vec_from_i32(context, 0)
}

fn one(context: &z3::Context) -> BitVec {
    bit_vec_from_i32(context, 1)
}

pub trait Component: Debug {
    fn arity(&self) -> usize;

    fn make_operator(&self, immediates: &[i32], operands: &[Id]) -> Operator;

    fn make_expression<'a>(
        &self,
        context: &'a z3::Context,
        immediates: &[BitVec<'a>],
        operands: &[BitVec<'a>],
        output: &BitVec<'a>,
    ) -> Bool<'a>;

    /// How many immediates does this component require?
    fn immediates(&self) -> usize {
        0
    }
}

pub fn const_(val: Option<i32>) -> Box<dyn Component> {
    #[derive(Debug)]
    struct Const(Option<i32>);

    impl Component for Const {
        fn arity(&self) -> usize {
            0
        }

        fn make_operator(&self, immediates: &[i32], _operands: &[Id]) -> Operator {
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
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            if let Some(val) = self.0 {
                BitVec::from_i64(context, val as i64, 32)._eq(output)
            } else {
                immediates[0]._eq(output)
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

    Box::new(Const(val)) as _
}

pub fn eqz() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Eqz;

    impl Component for Eqz {
        fn arity(&self) -> usize {
            1
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Eqz(operands[0])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            zero(context)
                ._eq(&operands[0])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Eqz) as _
}

pub fn clz() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Clz;

    impl Component for Clz {
        fn arity(&self) -> usize {
            1
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Clz(operands[0])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            fn clz<'a>(
                context: &'a z3::Context,
                input: &BitVec<'a>,
                zero_bit: &BitVec<'a>,
                i: u32,
            ) -> BitVec<'a> {
                if i == 32 {
                    bit_vec_from_i32(context, 32)
                } else {
                    input.extract(i, i)._eq(&zero_bit).ite(
                        &bit_vec_from_i32(context, i as i32),
                        &clz(context, input, zero_bit, i + 1),
                    )
                }
            }

            let zero_bit = BitVec::from_i64(context, 0, 1);
            clz(context, &operands[0], &zero_bit, 0)._eq(&output)
        }
    }

    Box::new(Clz) as _
}

pub fn ctz() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Ctz;

    impl Component for Ctz {
        fn arity(&self) -> usize {
            1
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Ctz(operands[0])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            fn ctz<'a>(
                context: &'a z3::Context,
                input: &BitVec<'a>,
                zero_bit: &BitVec<'a>,
                i: u32,
            ) -> BitVec<'a> {
                if i == 0 {
                    bit_vec_from_i32(context, 32)
                } else {
                    input.extract(31 - i, 31 - i)._eq(&zero_bit).ite(
                        &bit_vec_from_i32(context, i as i32),
                        &ctz(context, input, zero_bit, i + 1),
                    )
                }
            }

            let zero_bit = BitVec::from_i64(context, 0, 1);
            ctz(context, &operands[0], &zero_bit, 0)._eq(&output)
        }
    }

    Box::new(Ctz) as _
}

pub fn popcnt() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Popcnt;

    impl Component for Popcnt {
        fn arity(&self) -> usize {
            1
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Popcnt(operands[0])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            let mut bits: Vec<_> = (0..32)
                .map(|i| operands[0].extract(i, i).zero_ext(31))
                .collect();
            let initial = bits.pop().unwrap();
            bits.iter().fold(initial, |a, b| a.bvadd(b))._eq(&output)
        }
    }

    Box::new(Popcnt) as _
}

pub fn eq() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Eq;

    impl Component for Eq {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Eq(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                ._eq(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Eq) as _
}

pub fn ne() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Ne;

    impl Component for Ne {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Ne(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                ._eq(&operands[1])
                .ite(&zero(context), &one(context))
                ._eq(&output)
        }
    }

    Box::new(Ne) as _
}

pub fn lt_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Lts;

    impl Component for Lts {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::LtS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvslt(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Lts) as _
}

pub fn lt_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Ltu;

    impl Component for Ltu {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::LtU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvult(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Ltu) as _
}

pub fn gt_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Gts;

    impl Component for Gts {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::GtS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvsgt(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Gts) as _
}

pub fn gt_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Gtu;

    impl Component for Gtu {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::GtU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvugt(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Gtu) as _
}

pub fn le_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Les;

    impl Component for Les {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::LeS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvsle(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Les) as _
}

pub fn le_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Leu;

    impl Component for Leu {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::LeU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvule(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Leu) as _
}

pub fn ge_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Ges;

    impl Component for Ges {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::GeS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvsge(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Ges) as _
}

pub fn ge_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Geu;

    impl Component for Geu {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::GeU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvuge(&operands[1])
                .ite(&one(context), &zero(context))
                ._eq(&output)
        }
    }

    Box::new(Geu) as _
}

pub fn add() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Add;

    impl Component for Add {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Add(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvadd(&operands[1])._eq(output)
        }
    }

    Box::new(Add) as _
}

pub fn sub() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Sub;

    impl Component for Sub {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Sub(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvsub(&operands[1])._eq(output)
        }
    }

    Box::new(Sub) as _
}

pub fn mul() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Mul;

    impl Component for Mul {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Mul(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvmul(&operands[1])._eq(output)
        }
    }

    Box::new(Mul) as _
}

pub fn div_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Divs;

    impl Component for Divs {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::DivS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvsdiv(&operands[1])
                ._eq(&output)
                .and(&[&operands[1]._eq(&zero(context)).not()])
        }
    }

    Box::new(Divs) as _
}

pub fn div_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Divu;

    impl Component for Divu {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::DivU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvudiv(&operands[1])
                ._eq(&output)
                .and(&[&operands[1]._eq(&zero(context)).not()])
        }
    }

    Box::new(Divu) as _
}

pub fn rem_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Rems;

    impl Component for Rems {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::RemS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvsrem(&operands[1])
                ._eq(&output)
                .and(&[&operands[1]._eq(&zero(context)).not()])
        }
    }

    Box::new(Rems) as _
}

pub fn rem_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Remu;

    impl Component for Remu {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::RemU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0]
                .bvurem(&operands[1])
                ._eq(&output)
                .and(&[&operands[1]._eq(&zero(context)).not()])
        }
    }

    Box::new(Remu) as _
}

pub fn and() -> Box<dyn Component> {
    #[derive(Debug)]
    struct And;

    impl Component for And {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::And(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvand(&operands[1])._eq(output)
        }
    }

    Box::new(And) as _
}

pub fn or() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Or;

    impl Component for Or {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Or(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvor(&operands[1])._eq(output)
        }
    }

    Box::new(Or) as _
}

pub fn xor() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Xor;

    impl Component for Xor {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Xor(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvxor(&operands[1])._eq(output)
        }
    }

    Box::new(Xor) as _
}

pub fn shl() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Shl;

    impl Component for Shl {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Shl(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvshl(&operands[1])._eq(output)
        }
    }

    Box::new(Shl) as _
}

pub fn shr_s() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Shrs;

    impl Component for Shrs {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::ShrS(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvashr(&operands[1])._eq(output)
        }
    }

    Box::new(Shrs) as _
}

pub fn shr_u() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Shru;

    impl Component for Shru {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::ShrU(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvlshr(&operands[1])._eq(output)
        }
    }

    Box::new(Shru) as _
}

pub fn rotl() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Rotl;

    impl Component for Rotl {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Rotl(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvrotl(&operands[1])._eq(output)
        }
    }

    Box::new(Rotl) as _
}

pub fn rotr() -> Box<dyn Component> {
    #[derive(Debug)]
    struct Rotr;

    impl Component for Rotr {
        fn arity(&self) -> usize {
            2
        }

        fn make_operator(&self, _immediates: &[i32], operands: &[Id]) -> Operator {
            Operator::Rotr(operands[0], operands[1])
        }

        fn make_expression<'a>(
            &self,
            _context: &'a z3::Context,
            _immediates: &[BitVec<'a>],
            operands: &[BitVec<'a>],
            output: &BitVec<'a>,
        ) -> Bool<'a> {
            operands[0].bvrotr(&operands[1])._eq(output)
        }
    }

    Box::new(Rotr) as _
}
