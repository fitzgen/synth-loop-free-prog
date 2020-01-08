use crate::{Id, Instruction, Operator, Program};

#[derive(Debug)]
pub struct ProgramBuilder {
    program: Program,
}

impl ProgramBuilder {
    pub fn new() -> ProgramBuilder {
        ProgramBuilder {
            program: Program {
                instructions: vec![],
            },
        }
    }

    pub fn finish(self) -> Program {
        self.program
    }

    fn next_id(&self) -> Id {
        Id(self.program.instructions.len() as u32)
    }

    pub fn var(&mut self) -> Id {
        assert!(
            self.program
                .instructions
                .iter()
                .all(|inst| inst.operator == Operator::Var),
            "All `var`s must be at the start of the program"
        );

        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Var,
        });
        result
    }

    pub fn const_(&mut self, c: u64) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Const(c),
        });
        result
    }

    pub fn eqz(&mut self, a: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Eqz(a),
        });
        result
    }

    pub fn clz(&mut self, a: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Clz(a),
        });
        result
    }

    pub fn ctz(&mut self, a: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Ctz(a),
        });
        result
    }

    pub fn popcnt(&mut self, a: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Popcnt(a),
        });
        result
    }

    pub fn eq(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Eq(a, b),
        });
        result
    }

    pub fn ne(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Ne(a, b),
        });
        result
    }

    pub fn lt_s(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::LtS(a, b),
        });
        result
    }

    pub fn lt_u(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::LtU(a, b),
        });
        result
    }

    pub fn gt_s(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::GtS(a, b),
        });
        result
    }

    pub fn gt_u(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::GtU(a, b),
        });
        result
    }

    pub fn le_s(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::LeS(a, b),
        });
        result
    }

    pub fn le_u(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::LeU(a, b),
        });
        result
    }

    pub fn ge_s(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::GeS(a, b),
        });
        result
    }

    pub fn ge_u(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::GeU(a, b),
        });
        result
    }

    pub fn add(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Add(a, b),
        });
        result
    }

    pub fn sub(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Sub(a, b),
        });
        result
    }

    pub fn mul(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Mul(a, b),
        });
        result
    }

    pub fn div_s(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::DivS(a, b),
        });
        result
    }

    pub fn div_u(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::DivU(a, b),
        });
        result
    }

    pub fn rems(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::RemS(a, b),
        });
        result
    }

    pub fn remu(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::RemU(a, b),
        });
        result
    }

    pub fn and(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::And(a, b),
        });
        result
    }

    pub fn or(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Or(a, b),
        });
        result
    }

    pub fn xor(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Xor(a, b),
        });
        result
    }

    pub fn shl(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Shl(a, b),
        });
        result
    }

    pub fn shr_s(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::ShrS(a, b),
        });
        result
    }

    pub fn shr_u(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::ShrU(a, b),
        });
        result
    }

    pub fn rotl(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Rotl(a, b),
        });
        result
    }

    pub fn rotr(&mut self, a: Id, b: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Rotr(a, b),
        });
        result
    }

    pub fn select(&mut self, a: Id, b: Id, c: Id) -> Id {
        let result = self.next_id();
        self.program.instructions.push(Instruction {
            result,
            operator: Operator::Select(a, b, c),
        });
        result
    }
}
