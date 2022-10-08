use std::cell::RefCell;
use std::collections::{HashMap};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use crate::ast::*;

pub type RValue = Literal;

impl RValue {
    fn get_type(&self) -> Type {
        match self {
            RValue::Int(_) => Type::Int,
            RValue::Void => Type::Void,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum VariableMemory {
    Int(Vec<i32>),
}

pub type VariableMemoryRc = Rc<RefCell<VariableMemory>>;


#[derive(Clone, PartialEq, Debug)]
pub struct LValue {
    mem: VariableMemoryRc,
    idx: usize,
}

macro_rules! enum_cast {
    ($target: expr, $pat: path) => {
        {
            if let $pat(a) = $target { // #1
                a
            } else {
                panic!(
                    "mismatch variant when cast to {}",
                    stringify!($pat)); // #2
            }
        }
    };
}

impl LValue {
    fn get_type(&self) -> Type {
        match self.mem.borrow().deref() {
            VariableMemory::Int(_) => Type::Int,
        }
    }

    fn to_rvalue(&self) -> RValue {
        match self.mem.borrow().deref() {
            VariableMemory::Int(vec) => RValue::Int(vec[self.idx]),
        }
    }

    fn assign(&mut self, value: RValue) {
        match self.mem.borrow_mut().deref_mut() {
            // TODO static cast value to right type
            VariableMemory::Int(vec) => { vec[self.idx] = enum_cast!(value, RValue::Int) }
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    RValue(RValue),
    LValue(LValue),
}

impl Value {
    pub fn get_type(&self) -> Type {
        match self {
            Value::RValue(rvalue) => rvalue.get_type(),
            Value::LValue(lvalue) => lvalue.get_type(),
        }
    }

    pub fn to_rvalue(&self) -> RValue {
        match self {
            Value::RValue(rvalue) => *rvalue,
            Value::LValue(lvalue) => lvalue.to_rvalue(),
        }
    }
}


struct InterpreterMemory {
    // globals: HashMap<String, TypedLValue>,
    // each top entry represent variables from inside a function call layer
    // each entry inside a function layer represents the scope inside the function
    stack: Vec<Vec<HashMap<String, LValue>>>,

    all_inst: HashMap<usize, VariableMemoryRc>,
}

impl InterpreterMemory {
    pub fn new() -> InterpreterMemory {
        return InterpreterMemory { /*globals: HashMap::new(), */stack: Vec::new(), all_inst: HashMap::new() };
    }

    pub fn push_layer(&mut self) {
        self.stack.push(Vec::new());
    }

    pub fn pop_layer(&mut self) {
        assert_eq!(self.top_layer().len(), 0, "Expected to have no open scopes left for popping layer");
        self.stack.pop().unwrap();
    }

    pub fn top_layer(&mut self) -> &mut Vec<HashMap<String, LValue>> {
        return self.stack.last_mut().unwrap();
    }

    fn top_scope(&mut self) -> &mut HashMap<String, LValue> {
        return self.top_layer().last_mut().unwrap();
    }

    pub fn push_scope(&mut self) {
        self.top_layer().push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.top_layer().pop().unwrap();
    }

    pub fn add_variable(&mut self, name: String, value: RValue) {
        let mem = match value {
            RValue::Int(val) => {
                VariableMemoryRc::new(RefCell::new(VariableMemory::Int(vec![val])))
            }
            RValue::Void => panic!("Cannot instantiate void variable!"),
        };

        self.all_inst.insert(Rc::as_ptr(&mem) as usize, mem.clone());

        let scope = self.top_scope();

        if scope.contains_key(&*name) {
            panic!("Variable {} already declared", name);
        }

        scope.insert(name.clone(), LValue { mem, idx: 0 });
    }

    pub fn set_variable(&mut self, name: &String, value: RValue) {
        let var = self.lookup_value(name);
        var.assign(value);
    }

    pub fn lookup_value(&mut self, name: &String) -> &mut LValue {
        let top_layer = self.top_layer();

        for set in top_layer.iter_mut() {
            let val = set.get_mut(name);
            if val.is_some() {
                return val.unwrap();
            }
        }

        // TODO check whats wrong
        // return self.globals.get_mut(name).unwrap();
        panic!("Cannot find variable {}", name);
    }
}

pub struct Interpreter {
    variables: InterpreterMemory,
    translation_unit: Rc<TranslationUnit>,
}

macro_rules! enum_check {
    ($target: expr, $pat: path) => {
        {
            if let $pat(_) = $target { // #1
                $target
            } else {
                panic!(
                    "mismatch variant when cast to {}",
                    stringify!($pat)); // #2
            }
        }
    };
}

impl Interpreter {
    pub fn new(unit: TranslationUnit) -> Interpreter {
        Interpreter { variables: InterpreterMemory::new(), translation_unit: Rc::new(unit) }
    }

    pub fn empty() -> Interpreter {
        Interpreter { variables: InterpreterMemory::new(), translation_unit: Rc::new(TranslationUnit::new()) }
    }

    fn promote_operands(&mut self, lhs: Value, rhs: Value) -> (Value, Value) {
        return (lhs, rhs);
    }

    fn op_add(&mut self, lhs: Value, rhs: Value) -> Value {
        match lhs.to_rvalue() {
            RValue::Int(lhs) => { Value::RValue(RValue::Int(lhs + enum_cast!(rhs.to_rvalue(), RValue::Int))) }
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_sub(&mut self, lhs: Value, rhs: Value) -> Value {
        match lhs.to_rvalue() {
            RValue::Int(lhs) => { Value::RValue(RValue::Int(lhs - enum_cast!(rhs.to_rvalue(), RValue::Int))) }
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_mul(&mut self, lhs: Value, rhs: Value) -> Value {
        match lhs.to_rvalue() {
            RValue::Int(lhs) => { Value::RValue(RValue::Int(lhs * enum_cast!(rhs.to_rvalue(), RValue::Int))) }
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_div(&mut self, lhs: Value, rhs: Value) -> Value {
        match lhs.to_rvalue() {
            RValue::Int(lhs) => { Value::RValue(RValue::Int(lhs / enum_cast!(rhs.to_rvalue(), RValue::Int))) }
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_neq(&mut self, lhs: Value, rhs: Value) -> Value {
        match lhs.to_rvalue() {
            RValue::Int(lhs) => { Value::RValue(RValue::Int(if lhs != enum_cast!(rhs.to_rvalue(), RValue::Int) { 1 } else { 0 })) }
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_gt(&mut self, lhs: Value, rhs: Value) -> Value {
        match lhs.to_rvalue() {
            RValue::Int(lhs) => { Value::RValue(RValue::Int(if lhs > enum_cast!(rhs.to_rvalue(), RValue::Int) { 1 } else { 0 })) }
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }
}

impl Visitor<Value, Value> for Interpreter {
    fn visit_literal(&mut self, e: &Literal) -> Result<Value, Value> {
        return Ok(Value::RValue(*e));
    }

    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> Result<Value, Value> {
        if *op == Operator::Eq {
            match lhs {
                Expr::Variable(ref var) => {
                    let value = self.visit_expr(rhs)?;
                    self.variables.set_variable(&var.name, value.to_rvalue());
                    return Ok(value);
                }
                _ => panic!("Cannot assign value to expression")
            }
        } else {
            let lhs = self.visit_expr(lhs)?;
            let rhs = self.visit_expr(rhs)?;
            let (lhs, rhs) = self.promote_operands(lhs, rhs);

            match op {
                Operator::Plus => Ok(self.op_add(lhs, rhs)),
                Operator::Minus => Ok(self.op_sub(lhs, rhs)),
                Operator::Times => Ok(self.op_mul(lhs, rhs)),
                Operator::Div => Ok(self.op_div(lhs, rhs)),
                Operator::Neq => Ok(self.op_neq(lhs, rhs)),
                Operator::Greater => Ok(self.op_gt(lhs, rhs)),
                _ => panic!("Unsupported op {:?}", *op),
            }
        }
    }

    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> Result<Value, Value> {
        let value = match t {
            Type::Int => opt_init.as_ref().map_or(RValue::Int(0),
                                                  |expr| enum_check!(self.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int)),
            Type::Void => panic!("Cannot declare variable {} with type void.", name)
        };

        self.variables.add_variable(name.clone(), value);
        return Ok(Value::RValue(value));
    }

    fn visit_var(&mut self, var: &Variable) -> Result<Value, Value> {
        return Ok(Value::LValue(self.variables.lookup_value(&var.name).clone()));
    }

    fn visit_while(&mut self, cond: &Expr, body: &Stmt) -> Result<Value, Value> {
        while enum_cast!(self.visit_expr(cond)?.to_rvalue(), RValue::Int) != 0 {
            self.visit_statement(body)?;
        }

        return Ok(Value::RValue(RValue::Void));
    }

    fn visit_if(&mut self, cond: &Expr, body_if: &Stmt, opt_body_else: &Option<Box<Stmt>>) -> Result<Value, Value> {
        if enum_cast!(self.visit_expr(cond).unwrap().to_rvalue(), RValue::Int) != 0 {
            self.visit_statement(body_if)?;
            Ok(Value::RValue(RValue::Void))
        } else {
            opt_body_else.as_ref().map_or(Ok(Value::RValue(RValue::Void)), |body_else| {
                self.visit_statement(&*body_else)
            })
        }
    }

    fn visit_empty(&mut self) -> Result<Value, Value> {
        return Ok(Value::RValue(RValue::Void));
    }

    fn visit_function_call(&mut self, name: &str, args: &Vec<Value>) -> Result<Value, Value> {
        if name == "print" {
            for value in args.iter() {
                print!("{} ", enum_cast!(value.to_rvalue(), RValue::Int));
            }
            println!();

            return Ok(Value::RValue(RValue::Void));
        }


        let translation_unit = Rc::clone(&self.translation_unit);
        let func = translation_unit.functions.get(name).unwrap();

        if args.len() != func.params.len() {
            panic!("Function '{}' is called with {} arguments, while declared with {}", name, args.len(), func.params.len());
        }

        let mut variables: HashMap<String, Value> = HashMap::new();

        self.variables.push_layer();
        self.variables.push_scope();

        for idx in 0..args.len() {
            variables.insert(func.params.get(idx).unwrap().name.clone(), args.get(idx).unwrap().clone());
        }
        let result = self.visit_compound_statement(&func.body, &variables);
        self.variables.pop_scope();
        self.variables.pop_layer();

        match result {
            Ok(_) => {
                assert_eq!(func.ret_type, Type::Void, "A call to '{}' did not return a value, while return type is not void: {:?}", name, func.ret_type);
                Ok(Value::RValue(RValue::Void))
            }
            Err(r) =>
                {
                    let res_type = r.get_type();
                    // TODO: cast to return type instead
                    assert_eq!(func.ret_type, res_type, "Function '{}' return value of type {:?} while expected {:?}.", name, res_type, func.ret_type);
                    Ok(r)
                }
        }
    }

    fn visit_compound_statement(&mut self, compound: &CompoundStmt, variables: &HashMap<String, Value>) -> Result<Value, Value> {
        self.variables.push_scope();

        for (name, value) in variables {
            self.variables.add_variable(name.clone(), value.to_rvalue());
        }

        let mut func = || -> Result<Value, Value> {
            for stmt in &compound.statements {
                self.visit_statement(stmt)?;
            }

            Ok(Value::RValue(RValue::Void))
        };

        let result = func();
        self.variables.pop_scope();
        return result;
    }

    fn visit_return(&mut self, expr: &Expr) -> Result<Value, Value> {
        return Err(self.visit_expr(expr).unwrap());
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::interpreter::*;
    use crate::{parser};

    #[test]
    fn visit_literal() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::Literal(Literal::Int(42));
        assert_eq!(interpreter.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int(42));
    }

    #[test]
    fn visit_binary_plus() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal::Int(3))),
                                    Box::new(Expr::Literal(Literal::Int(5))),
                                    Operator::Plus);
        assert_eq!(interpreter.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int(8));
    }

    #[test]
    fn visit_binary_minus() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal::Int(3))),
                                    Box::new(Expr::Literal(Literal::Int(5))),
                                    Operator::Minus);
        assert_eq!(interpreter.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int(-2));
    }

    #[test]
    fn visit_binary_times() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal::Int(3))),
                                    Box::new(Expr::Literal(Literal::Int(5))),
                                    Operator::Times);
        assert_eq!(interpreter.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int(15));
    }

    #[test]
    fn visit_binary_div() {
        let mut interpreter = Interpreter::empty();
        let expr = Expr::BinaryExpr(Box::new(Expr::Literal(Literal::Int(39))),
                                    Box::new(Expr::Literal(Literal::Int(5))),
                                    Operator::Div);
        assert_eq!(interpreter.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int(7));
    }

    #[test]
    fn integration() {
        let tokens = parser::tokenize("(5+3)-(2)*7");
        let expr = parser::parse_expr(tokens);

        let mut interp = Interpreter::empty();
        let result = interp.visit_expr(&expr);

        assert_eq!(result.unwrap().to_rvalue(), RValue::Int(-6));
    }
}

