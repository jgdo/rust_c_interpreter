use std::collections::HashMap;
use crate::ast::*;

struct VariableMemory {
    // TODO this is actually false, we need a global set of variables, and a stack of variable sets for function calls
    variables_set: Vec<HashMap<String, i32>>,
}

impl VariableMemory {
    pub fn new() -> VariableMemory {
        return VariableMemory { variables_set: Vec::new() };
    }

    fn top_scope(&mut self) -> &mut HashMap<String, i32> {
        return self.variables_set.last_mut().unwrap();
    }

    pub fn push_scope(&mut self) {
        self.variables_set.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.variables_set.pop().unwrap();
    }

    pub fn add_variable(&mut self, name: String, value: i32) {
        let mut scope = self.top_scope();

        if scope.contains_key(&*name) {
            panic!("Variable {} already declared", name);
        }

        scope.insert(name.clone(), value);
    }

    pub fn set_variable(&mut self, name: &String, value: i32) {
        let mut var = self.lookup_value(name);
        *var = value;
    }

    pub fn lookup_value(&mut self, name: &String) -> &mut i32 {
        for set in self.variables_set.iter_mut() {
            let val = set.get_mut(name);
            if val.is_some() {
                return val.unwrap();
            }
        }

        panic!("Variable {} not found", name);
    }
}

pub struct Interpreter {
    variables: VariableMemory,
}

impl Interpreter {
    pub(crate) fn new() -> Interpreter {
        Interpreter { variables: VariableMemory::new() }
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
                        self.variables.set_variable(&var.name, value);
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
        let value = opt_init.as_ref().map_or(0, |expr| self.visit_expr(&expr));
        self.variables.add_variable(name.clone(), value);
        return value;
    }

    fn visit_var(&mut self, var: &Variable) -> i32 {
        return *self.variables.lookup_value(&var.name);
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

    fn visit_compound_statement(&mut self, compound: &CompoundStmt, variables: &HashMap<String, i32>) -> i32 {
        self.variables.push_scope();

        for (name, value) in variables {
            self.variables.add_variable(name.clone(), *value);
        }

        let (last, first) = compound.statements.split_last().unwrap();

        for stmt in first {
            self.visit_statement(stmt);
        }
        let result = self.visit_statement(last);

        self.variables.pop_scope();
        return result;
    }

    fn visit_function_call(&mut self, unit: &TranslationUnit, name: &str, args: &Vec<i32>) -> i32 {
        let func = unit.functions.get(name).unwrap();
        assert_eq!(args.len(), func.params.len());

        let mut variables: HashMap<String, i32> = HashMap::new();

        self.variables.push_scope();
        for idx in 0..args.len() {
            variables.insert(func.params.get(idx).unwrap().name.clone(), *args.get(idx).unwrap());
        }
        let result = self.visit_compound_statement(&func.body, &variables);
        self.variables.pop_scope();

        return result;
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

