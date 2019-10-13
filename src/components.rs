use crate::{Component, Id, Operator, Specification};
use z3::ast::{Ast, BV as BitVec};

fn fresh_input(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "input", 32)
}

fn fresh_output(context: &z3::Context) -> BitVec {
    BitVec::fresh_const(context, "output", 32)
}

fn bit_vec_from_i32(context: &z3::Context, val: i32) -> BitVec {
    BitVec::from_i64(context, val as i64, 32)
}

fn zero(context: &z3::Context) -> BitVec {
    bit_vec_from_i32(context, 0)
}

fn one(context: &z3::Context) -> BitVec {
    bit_vec_from_i32(context, 1)
}

impl<'a> Component<'a> {
    pub fn const_(context: &'a z3::Context) -> Self {
        let inputs = vec![];
        let output = fresh_output(context);
        let expression = BitVec::fresh_const(context, "const", 32)._eq(&output);
        Component {
            operator: Operator::Const(0),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn eqz(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context)];
        let output = fresh_output(context);
        let expression = zero(context)
            ._eq(&inputs[0])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::Eqz(Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn clz(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context)];
        let output = fresh_output(context);

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
        let expression = clz(context, &inputs[0], &zero_bit, 0)._eq(&output);
        Component {
            operator: Operator::Clz(Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn ctz(context: &'a z3::Context) -> Self {
        let inputs = vec![];
        let output = fresh_output(context);

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
        let expression = ctz(context, &inputs[0], &zero_bit, 0)._eq(&output);
        Component {
            operator: Operator::Ctz(Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn popcnt(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context)];
        let output = fresh_output(context);
        let mut bits: Vec<_> = (0..32)
            .map(|i| inputs[0].extract(i, i).zero_ext(31))
            .collect();
        let initial = bits.pop().unwrap();
        let expression = bits.iter().fold(initial, |a, b| a.bvadd(b))._eq(&output);
        Component {
            operator: Operator::Popcnt(Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn eq(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            ._eq(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::Eq(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn ne(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            ._eq(&inputs[1])
            .ite(&zero(context), &one(context))
            ._eq(&output);
        Component {
            operator: Operator::Ne(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn lts(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvslt(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::LtS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn ltu(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvult(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::LtU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn gts(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvsgt(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::GtS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn gtu(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvugt(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::GtU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn les(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvsle(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::LeS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn leu(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvule(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::LeU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn ges(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvsge(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::GeS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn geu(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvuge(&inputs[1])
            .ite(&one(context), &zero(context))
            ._eq(&output);
        Component {
            operator: Operator::GeU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn add(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvadd(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Add(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn sub(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvsub(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Sub(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn mul(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvmul(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Mul(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn divs(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvsdiv(&inputs[1])
            ._eq(&output)
            .and(&[&inputs[1]._eq(&zero(context)).not()]);
        Component {
            operator: Operator::DivS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn divu(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvudiv(&inputs[1])
            ._eq(&output)
            .and(&[&inputs[1]._eq(&zero(context)).not()]);
        Component {
            operator: Operator::DivU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn rems(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvsrem(&inputs[1])
            ._eq(&output)
            .and(&[&inputs[1]._eq(&zero(context)).not()]);
        Component {
            operator: Operator::RemS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn remu(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0]
            .bvurem(&inputs[1])
            ._eq(&output)
            .and(&[&inputs[1]._eq(&zero(context)).not()]);
        Component {
            operator: Operator::RemU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn and(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvand(&inputs[1])._eq(&output);
        Component {
            operator: Operator::And(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn or(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvor(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Or(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn xor(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvxor(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Xor(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn shl(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvshl(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Shl(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn shrs(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvashr(&inputs[1])._eq(&output);
        Component {
            operator: Operator::ShrS(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn shru(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvlshr(&inputs[1])._eq(&output);
        Component {
            operator: Operator::ShrU(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn rotl(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvrotl(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Rotl(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }

    pub fn rotr(context: &'a z3::Context) -> Self {
        let inputs = vec![fresh_input(context), fresh_input(context)];
        let output = fresh_output(context);
        let expression = inputs[0].bvrotr(&inputs[1])._eq(&output);
        Component {
            operator: Operator::Rotr(Id(0), Id(0)),
            spec: Specification {
                inputs,
                output,
                expression,
            },
        }
    }
}
