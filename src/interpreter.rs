use std::cell::{RefCell};
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
            RValue::Ptr(t, _, _) => Type::Ptr(Box::new(t.clone())),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum VariableMemory {
    Int(Vec<i32>),
    Ptr(Type, Vec<(usize, usize)>), // all inner types will be same
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

// https://stackoverflow.com/questions/54035728/how-to-add-a-negative-i32-number-to-an-usize-variable
fn add(u: usize, i: i32) -> usize {
    if i.is_negative() {
        u - i.wrapping_abs() as u32 as usize
    } else {
        u + i as usize
    }
}

impl LValue {
    fn get_type(&self) -> Type {
        match self.mem.borrow().deref() {
            VariableMemory::Int(_) => Type::Int,
            VariableMemory::Ptr(t, _) => Type::Ptr(Box::new(t.clone())),
        }
    }

    fn to_rvalue(&self) -> RValue {
        match self.mem.borrow().deref() {
            VariableMemory::Int(vec) => RValue::Int(vec[self.idx]),
            VariableMemory::Ptr(t, vec) => RValue::Ptr(t.clone(), vec[self.idx].0, vec[self.idx].1)
        }
    }

    fn assign(&mut self, value: RValue) {
        match self.mem.borrow_mut().deref_mut() {
            // TODO static cast value to right type
            VariableMemory::Int(vec) => { vec[self.idx] = enum_cast!(value, RValue::Int) }
            VariableMemory::Ptr(t, vec) => {
                let (value_t, value_hash, value_idx) = match value {
                    RValue::Ptr(t, h, i) => (t, h, i),
                    _ => panic!("type does not match"),
                };

                assert_eq!(*t, value_t, "Trying to assign a pointer to a different type of pointer");
                vec[self.idx] = (value_hash, value_idx);
            }
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
            Value::RValue(rvalue) => rvalue.clone(),
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

    pub fn add_variable(&mut self, name: String, t: &Type, init_value: RValue) {
        /*

        let value = match t {
            Type::Int => opt_init.as_ref().map_or(RValue::Int(0),
                                                  |expr| enum_check!(self.visit_expr(&expr).unwrap().to_rvalue(), RValue::Int)),
            Type::Void => panic!("Cannot declare variable {} with type void.", name),
            Type::Ptr(t_var) => opt_init.as_ref().map_or(RValue::Ptr(t_var.deref().clone(), 0, 0),
                                                         |expr| {
                                                             match self.visit_expr(&expr).unwrap().to_rvalue() {
                                                                 RValue::Ptr(t_value, h, i) => {
                                                                     assert_eq!(t_var.deref().clone(), t_value, "trying to assign a pointer an address to wrong type");
                                                                     RValue::Ptr(t_value, h, i)
                                                                 }
                                                                 _ => panic!("trying to assign something to a pointer that isn't an address"),
                                                             }
                                                         }),
        };

         */

        /*
        let mem = match value {
            RValue::Int(val) => {
                VariableMemoryRc::new(RefCell::new(VariableMemory::Int(vec![val])))
            }
            RValue::Void => panic!("Cannot instantiate void variable!"),
            RValue::Ptr(t, hash, idx) => {
                VariableMemoryRc::new(RefCell::new(VariableMemory::Ptr(t.clone(), vec![(hash, idx)])))
            }
        };
        */

        // TODO properly promote init_value to t type
        let mem = match t {
            Type::Int => {
                VariableMemoryRc::new(RefCell::new(VariableMemory::Int(vec![enum_cast!(init_value, RValue::Int)])))
            },
            Type::Void => panic!("Cannot declare variable {} with type void.", name),
            Type::Ptr(t_ptr) => {
                match init_value {
                    RValue::Ptr(t, hash, idx) => VariableMemoryRc::new(RefCell::new(VariableMemory::Ptr(t.clone(), vec![(hash, idx)]))),
                    _ => panic!("trying to assign something to a pointer that isn't an address")
                }
            }
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

    pub fn lookup_ptr(&mut self, hash: usize, idx: usize) -> LValue
    {
        LValue { mem: self.all_inst.get_mut(&hash).unwrap().clone(), idx }
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
        // TODO: do a real promotion!
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
        return Ok(Value::RValue(e.clone()));
    }

    fn visit_unary_expr(&mut self, expr: &Expr, op: &Operator) -> Result<Value, Value> {
        match op {
            Operator::Times => {
                let ptr = self.visit_expr(expr)?.to_rvalue();
                match ptr {
                    RValue::Ptr(t, hash, idx) => {
                        let value = self.variables.lookup_ptr(hash, idx);
                        // TODO check if type of value equals type of t
                        Ok(Value::LValue(value))
                    }
                    _ => panic!("Trying to dereference an expression that is no a pointer"),
                }
            }
            Operator::And => {
                let value = enum_cast!(self.visit_expr(expr)?, Value::LValue);
                return Ok(Value::RValue(RValue::Ptr(value.get_type(), Rc::as_ptr(&value.mem) as usize, value.idx)));
            }
            _ => panic!("Unknown unary operator {:?}", op)
        }
    }

    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> Result<Value, Value> {
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
            Operator::Eq => {
                let mut var = enum_cast!(lhs, Value::LValue);
                var.assign(rhs.to_rvalue());
                return Ok(Value::LValue(var));
            }
            _ => panic!("Unsupported op {:?}", *op),
        }
    }

    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> Result<Value, Value> {
        // TODO: make a table of default int values per type instead
        let value = match t {
            Type::Int => opt_init.as_ref().map_or(RValue::Int(0),
                                                  |expr| self.visit_expr(&expr).unwrap().to_rvalue()),
            Type::Void => panic!("Cannot declare variable {} with type void.", name),
            Type::Ptr(t_var) => opt_init.as_ref().map_or(RValue::Ptr(t_var.deref().clone(), 0, 0),
                                                         |expr| {  self.visit_expr(&expr).unwrap().to_rvalue()}),
        };

        self.variables.add_variable(name.clone(), t,value.clone());
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
            let r_value = value.to_rvalue();
            self.variables.add_variable(name.clone(), &r_value.get_type() ,r_value);
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

    fn visit_index_expr(&mut self, expr: &Expr, index_expr: &Expr) -> Result<Value, Value> {
        let expr_value = self.visit_expr(expr)?.to_rvalue();
        let index = enum_cast!(self.visit_expr(index_expr)?.to_rvalue(), RValue::Int);

        match expr_value {
            RValue::Ptr(_, h, i) =>
                Ok(Value::LValue(self.variables.lookup_ptr(h, add(i, index)))),
            _ => panic!("Cannot index non-pointer value"),
        }
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

