use std::collections::HashMap;
use std::f32::consts::E;
use std::rc::Rc;
use crate::ast::*;

struct VariableMemory {
    globals: HashMap<String, i32>,
    // each top entry represent variables from inside a function call layer
    // each entry inside a function layer represents the scope inside the function
    variables_set: Vec<Vec<HashMap<String, i32>>>,
}

impl VariableMemory {
    pub fn new() -> VariableMemory {
        return VariableMemory { globals: HashMap::new(), variables_set: Vec::new() };
    }

    pub fn push_layer(&mut self) {
        self.variables_set.push(Vec::new());
    }

    pub fn pop_layer(&mut self) {
        assert_eq!(self.top_layer().len(), 0, "Expected to have no open scopes left for popping layer");
        self.variables_set.pop().unwrap();
    }

    pub fn top_layer(&mut self) -> &mut Vec<HashMap<String, i32>> {
        return self.variables_set.last_mut().unwrap();
    }

    fn top_scope(&mut self) -> &mut HashMap<String, i32> {
        return self.top_layer().last_mut().unwrap();
    }

    pub fn push_scope(&mut self) {
        self.top_layer().push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.top_layer().pop().unwrap();
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
        let top_layer = self.top_layer();

        for set in top_layer.iter_mut() {
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
    translation_unit: Rc<TranslationUnit>,
}

impl Interpreter {
    pub fn new(unit: TranslationUnit) -> Interpreter {
        Interpreter { variables: VariableMemory::new(), translation_unit: Rc::new(unit) }
    }

    pub fn empty() -> Interpreter {
        Interpreter { variables: VariableMemory::new(), translation_unit: Rc::new(TranslationUnit::new()) }
    }
}

impl Visitor<i32, i32> for Interpreter {
    fn visit_literal(&mut self, e: &Literal) -> Result<i32, i32> {
        return Ok(e.value);
    }

    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> Result<i32, i32> {
        match op {
            Operator::Plus => Ok(self.visit_expr(&lhs)? + self.visit_expr(&rhs)?),
            Operator::Minus => Ok(self.visit_expr(lhs)? - self.visit_expr(rhs)?),
            Operator::Times => Ok(self.visit_expr(lhs)? * self.visit_expr(rhs)?),
            Operator::Div => Ok(self.visit_expr(lhs)? / self.visit_expr(rhs)?),
            Operator::Eq => {
                match lhs {
                    Expr::Variable(ref var) => {
                        let value = self.visit_expr(rhs)?;
                        self.variables.set_variable(&var.name, value);
                        return Ok(value);
                    }
                    _ => panic!("Cannot assign value to expression")
                }
            }
            Operator::Neq => if self.visit_expr(lhs) != self.visit_expr(rhs) { Ok(1) } else { Ok(0) },
            Operator::Greater => if self.visit_expr(lhs) > self.visit_expr(rhs) { Ok(1) } else { Ok(0) },
        }
    }

    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> Result<i32, i32> {
        let value = opt_init.as_ref().map_or(0, |expr| self.visit_expr(&expr).unwrap());
        self.variables.add_variable(name.clone(), value);
        return Ok(value);
    }

    fn visit_var(&mut self, var: &Variable) -> Result<i32, i32> {
        return Ok(*self.variables.lookup_value(&var.name));
    }

    fn visit_while(&mut self, cond: &Expr, body: &Stmt) -> Result<i32, i32> {
        while self.visit_expr(cond)? != 0 {
            self.visit_statement(body);
        }

        return Ok(0); // TODO this is actually a hack
    }

    fn visit_if(&mut self, cond: &Expr, body_if: &Stmt, opt_body_else: &Option<Box<Stmt>>) -> Result<i32, i32> {
        if self.visit_expr(cond).unwrap() != 0 {
            self.visit_statement(body_if)?;
            Ok(0) // TODO this is actually a hack
        } else {
            opt_body_else.as_ref().map_or(Ok(0), |body_else| {
                self.visit_statement(&*body_else)
            })
        }
    }

    fn visit_empty(&mut self) -> Result<i32, i32> {
        return Ok(0); // TODO this is actually a hack
    }

    fn visit_compound_statement(&mut self, compound: &CompoundStmt, variables: &HashMap<String, i32>) -> Result<i32, i32> {
        self.variables.push_scope();

        for (name, value) in variables {
            self.variables.add_variable(name.clone(), *value);
        }

        let mut func = || -> Result<i32, i32> {
            for stmt in &compound.statements {
                self.visit_statement(stmt)?;
            }

            Ok(0)  // hack
        };

        let result = func();
        self.variables.pop_scope();
        return result;
    }

    fn visit_function_call(&mut self, name: &str, args: &Vec<i32>) -> Result<i32, i32> {
        if name == "print"{
            for value in args.iter() {
                print!("{} ", *value);
            }
            println!();

            return Ok(0)
        }


        let translation_unit = Rc::clone(&self.translation_unit);
        let func = translation_unit.functions.get(name).unwrap();

        if args.len() != func.params.len() {
            panic!("Function '{}' is called with {} arguments, while declared with {}", name, args.len(), func.params.len());
        }

        let mut variables: HashMap<String, i32> = HashMap::new();

        self.variables.push_layer();
        self.variables.push_scope();
        for idx in 0..args.len() {
            variables.insert(func.params.get(idx).unwrap().name.clone(), *args.get(idx).unwrap());
        }
        let result = self.visit_compound_statement(&func.body, &variables);
        self.variables.pop_scope();
        self.variables.pop_layer();

        match result {
            Ok(_) => panic!("A call to {} did not return a value", name),
            Err(r) => Ok(r)
        }
    }

    fn visit_return(&mut self, expr: &Expr) -> Result<i32, i32> {
        return Err(self.visit_expr(expr).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::interpreter::*;
    use crate::{interpreter, parser};

    #[test]
    fn visit_literal() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::Literal(Literal { value: 42 });
        assert_eq!(interpreter.visit_expr(&expr).unwrap(), 42);
    }

    #[test]
    fn visit_binary_plus() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Plus);
        assert_eq!(interpreter.visit_expr(&expr).unwrap(), 8);
    }

    #[test]
    fn visit_binary_minux() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Minus);
        assert_eq!(interpreter.visit_expr(&expr).unwrap(), -2);
    }

    #[test]
    fn visit_binary_times() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 3 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Times);
        assert_eq!(interpreter.visit_expr(&expr).unwrap(), 15);
    }

    #[test]
    fn visit_binary_div() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal { value: 39 })),
                                    Box::new(Expr::Literal(Literal { value: 5 })),
                                    Operator::Div);
        assert_eq!(interpreter.visit_expr(&expr).unwrap(), 7);
    }

    #[test]
    fn integration() {
        let tokens = parser::tokenize("(5+3)-(2)*7");
        let expr = parser::parse_expr(tokens);

        let mut interp = Interpreter::empty();
        let result = interp.visit_expr(&expr);

        assert_eq!(result.unwrap(), -6);
    }
}

