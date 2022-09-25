pub enum Operator {
    Plus,
    Minus,
    Times,
    Div,
}

pub struct Literal {
    value: i32,
}

pub enum Expr {
    Literal(Literal),
    BinaryExpr(Box<Expr>, Box<Expr>, Operator),
}


pub trait Visitor<R> {
    fn visit_literal(&self, e: &Literal) -> R;
    fn visit_binary_exp(&self, lhs: &Expr, rhs: &Expr, op: &Operator) -> R;

    fn visit_expr(&self, e: &Expr) -> R {
        match e {
            Expr::Literal(ref lit) => self.visit_literal(lit),
            Expr::BinaryExpr(ref lhs, ref rhs, ref op) => self.visit_binary_exp(lhs, rhs, op),
        }
    }
}

struct Interpreter;

impl Visitor<i32> for Interpreter {
    fn visit_literal(&self, e: &Literal) -> i32 {
        return e.value;
    }

    fn visit_binary_exp(&self, lhs: &Expr, rhs: &Expr, op: &Operator) -> i32 {
        match op {
            Operator::Plus => self.visit_expr(&lhs) + self.visit_expr(&rhs),
            Operator::Minus => self.visit_expr(lhs) - self.visit_expr(rhs),
            Operator::Times => self.visit_expr(lhs) * self.visit_expr(rhs),
            Operator::Div => self.visit_expr(lhs) / self.visit_expr(rhs),
        }
    }
}

fn main() {

}

#[cfg(test)]
mod tests {
    use crate::{Expr, Literal, Interpreter, Visitor, Operator};

    #[test]
    fn visit_literal() {
        let interpreter = Interpreter {};
        let expr = Expr::Literal(Literal { value: 42 });
        assert_eq!(interpreter.visit_expr(&expr), 42);
    }

    #[test]
    fn visit_binary() {
        let interpreter = Interpreter {};
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Plus);
        assert_eq!(interpreter.visit_expr(&expr), 8);
    }
}