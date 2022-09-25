use crate::ast::*;

pub struct Interpreter;

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

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::interpreter::*;

    #[test]
    fn visit_literal() {
        let interpreter = Interpreter {};
        let expr = Expr::Literal(Literal { value: 42 });
        assert_eq!(interpreter.visit_expr(&expr), 42);
    }

    #[test]
    fn visit_binary_plus() {
        let interpreter = Interpreter {};
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Plus);
        assert_eq!(interpreter.visit_expr(&expr), 8);
    }

    #[test]
    fn visit_binary_minux() {
        let interpreter = Interpreter {};
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Minus);
        assert_eq!(interpreter.visit_expr(&expr), -2);
    }

    #[test]
    fn visit_binary_times() {
        let interpreter = Interpreter {};
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Times);
        assert_eq!(interpreter.visit_expr(&expr), 15);
    }

    #[test]
    fn visit_binary_div() {
        let interpreter = Interpreter {};
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 39 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Div);
        assert_eq!(interpreter.visit_expr(&expr), 7);
    }
}

