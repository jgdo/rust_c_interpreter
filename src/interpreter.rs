use std::collections::HashMap;
use crate::ast::*;

pub struct Interpreter {
    variables: HashMap<String, i32>,
}

impl Interpreter {
    pub(crate) fn new() -> Interpreter {
        Interpreter { variables: HashMap::new() }
    }
}

impl Visitor<i32> for Interpreter {
    fn visit_literal(&mut self, e: &Literal) -> i32 {
        return e.value;
    }

    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> i32 {
        match op {
            Operator::Plus => self.visit_expr(&lhs) + self.visit_expr(&rhs),
            Operator::Minus => self.visit_expr(lhs) - self.visit_expr(rhs),
            Operator::Times => self.visit_expr(lhs) * self.visit_expr(rhs),
            Operator::Div => self.visit_expr(lhs) / self.visit_expr(rhs),
            Operator::Eq => {
                match lhs {
                    Expr::Variable(ref var) => {
                        let value = self.visit_expr(rhs);

                        self.variables.insert(var.name.clone(), value).unwrap();
                        return value;
                    }
                    _ => panic!("Cannot assign value to expression")
                }
            }
            Operator::Neq => if self.visit_expr(lhs) != self.visit_expr(rhs) { 1 } else { 0 },
            Operator::Greater => if self.visit_expr(lhs) > self.visit_expr(rhs) { 1 } else { 0 },
        }
    }

    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> i32 {
        if self.variables.contains_key(name) {
            panic!("Variable {} already declared", name);
        }

        let value = opt_init.as_ref().map_or(0, |expr| self.visit_expr(&expr));
        self.variables.insert(name.clone(), value);
        return value;
    }

    fn visit_var(&mut self, var: &Variable) -> i32 {
        *self.variables.get(&*var.name).unwrap()
    }

    fn visit_while(&mut self, cond: &Expr, body: &Stmt) -> i32 {
        while self.visit_expr(cond) != 0 {
            self.visit_statement(body);
        }

        return 0; // TODO this is actually a hack
    }

    fn visit_if(&mut self, cond: &Expr, body_if: &Stmt, opt_body_else: &Option<Box<Stmt>>) -> i32 {
        if self.visit_expr(cond) != 0 {
            self.visit_statement(body_if)
        } else {
            opt_body_else.as_ref().map_or(0, |body_else| { // TODO this is actually a hack
                self.visit_statement(&*body_else)
            })
        }
    }

    fn visit_empty(&mut self) -> i32 {
        return 0; // TODO this is actually a hack
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::interpreter::*;
    use crate::{interpreter, parser};

    #[test]
    fn visit_literal() {
        let mut interpreter = Interpreter::new();
        let expr = Expr::Literal(Literal { value: 42 });
        assert_eq!(interpreter.visit_expr(&expr), 42);
    }

    #[test]
    fn visit_binary_plus() {
        let mut interpreter = Interpreter::new();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Plus);
        assert_eq!(interpreter.visit_expr(&expr), 8);
    }

    #[test]
    fn visit_binary_minux() {
        let mut interpreter = Interpreter::new();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Minus);
        assert_eq!(interpreter.visit_expr(&expr), -2);
    }

    #[test]
    fn visit_binary_times() {
        let mut interpreter = Interpreter::new();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Times);
        assert_eq!(interpreter.visit_expr(&expr), 15);
    }

    #[test]
    fn visit_binary_div() {
        let mut interpreter = Interpreter::new();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 39 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Div);
        assert_eq!(interpreter.visit_expr(&expr), 7);
    }

    #[test]
    fn integration() {
        let tokens = parser::tokenize("(5+3)-(2)*7");
        let expr = parser::parse_expr(tokens);

        let mut interp = Interpreter::new();
        let result = interp.visit_expr(&expr);

        assert_eq!(result, -6);
    }
}

