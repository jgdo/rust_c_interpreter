use std::cell::{RefCell};
use std::collections::{HashMap};
use std::convert::TryFrom;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::fmt;
use crate::ast::*;

pub type RValue = Literal;

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Char => write!(f, "char"),
            Type::Void => write!(f, "void"),
            Type::Ptr(inner_type) => write!(f, "{}*", *inner_type),
            Type::Array(len, inner_type) => write!(f, "{}[{}]", **inner_type, len),
        }
    }
}

impl RValue {
    fn get_type(&self) -> Type {
        match self {
            RValue::Int(_) => Type::Int,
            RValue::Char(_) => Type::Char,
            RValue::Void => Type::Void,
            RValue::Ptr(t, _, _) => Type::create_ptr(t.clone()),
            RValue::Str(_) => Type::create_ptr(Type::Char),
        }
    }

    fn promote_to(&self, target_type: Type) -> RValue {
        match self {
            RValue::Int(val) => {
                match target_type {
                    Type::Int => return self.clone(),
                    Type::Char => return RValue::Char(std::char::from_u32(u32::try_from(*val).unwrap()).unwrap()),
                    _ => {}
                }
            }
            RValue::Char(val) => {
                match target_type {
                    Type::Int => return RValue::Int(u32::from(*val) as i32),
                    Type::Char => return self.clone(),
                    _ => {}
                }
            }
            RValue::Void => if target_type == Type::Void { return self.clone(); },
            RValue::Ptr(self_elem_type, _, _) =>
                if target_type == Type::create_ptr(self_elem_type.clone()) { return self.clone(); },
            // TODO RValue::Str(string) =>
            //    if target_type == Type::create_ptr(Type::Char) { return self.clone(); },
            _ => {}
        };

        panic!("Cannot convert {} to {}", self.get_type(), target_type);
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum VariableMemory {
    Int(Vec<i32>),
    Char(Vec<char>),
    Ptr(Type, Vec<(usize, usize)>), // element type
}

pub type VariableMemoryRc = Rc<RefCell<VariableMemory>>;


#[derive(Clone, PartialEq, Debug)]
pub struct LValue {
    mem: VariableMemoryRc,
    idx: Option<usize>, // index of the variable inside mem. If none, this LValue represents an array
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
            VariableMemory::Char(_) => Type::Char,
            VariableMemory::Ptr(element_type, _) => Type::create_ptr(element_type.clone()),
        }
    }

    fn to_rvalue(&self) -> RValue {
        match self.mem.borrow().deref() {
            VariableMemory::Int(vec) => {
                match self.idx {
                    Some(my_idx) => RValue::Int(vec[my_idx]),
                    None => RValue::Ptr(Type::Int, Rc::as_ptr(&self.mem) as usize, 0)
                }
            }
            VariableMemory::Char(vec) => {
                match self.idx {
                    Some(my_idx) => RValue::Char(vec[my_idx]),
                    None => RValue::Ptr(Type::Char, Rc::as_ptr(&self.mem) as usize, 0)
                }
            }
            VariableMemory::Ptr(t, vec) => RValue::Ptr(t.clone(), vec[self.idx.unwrap()].0, vec[self.idx.unwrap()].1)
        }
    }

    fn assign(&mut self, value: RValue) {
        match self.mem.borrow_mut().deref_mut() {
            VariableMemory::Int(vec) => vec[self.idx.unwrap()] = enum_cast!(value.promote_to(Type::Int), RValue::Int),
            VariableMemory::Char(vec) => vec[self.idx.unwrap()] = enum_cast!(value.promote_to(Type::Char), RValue::Char),
            VariableMemory::Ptr(t, vec) => {
                let (value_t, value_hash, value_idx) = match value {
                    RValue::Ptr(t, h, i) => (t, h, i),
                    _ => panic!("type does not match"),
                };

                assert_eq!(*t, value_t, "Trying to assign a pointer to a different type of pointer");
                vec[self.idx.unwrap()] = (value_hash, value_idx);
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

    pub fn add_variable(&mut self, name: String, var_type: &Type, init_value: RValue) {
        let mem = match var_type {
            Type::Int => {
                VariableMemoryRc::new(RefCell::new(VariableMemory::Int(vec![enum_cast!(init_value.promote_to(Type::Int), RValue::Int)])))
            }
            Type::Char => {
                VariableMemoryRc::new(RefCell::new(VariableMemory::Char(vec![enum_cast!(init_value.promote_to(Type::Char), RValue::Char)])))
            }
            Type::Void => panic!("Cannot declare variable {} with type void.", name),
            Type::Ptr(t_ptr) => {
                match init_value {
                    RValue::Ptr(t, hash, idx) => {
                        assert_eq!(**t_ptr, t, "Source pointer type does not match to target pointer type");
                        VariableMemoryRc::new(RefCell::new(VariableMemory::Ptr(t.clone(), vec![(hash, idx)])))
                    }
                    _ => panic!("trying to assign something to a pointer that isn't an address")
                }
            }
            Type::Array(array_len, elem_type_box) => {
                assert_eq!(**elem_type_box, Type::Int, "only int arrays supported");
                let init_elem = enum_cast!(init_value, RValue::Int);
                VariableMemoryRc::new(RefCell::new(VariableMemory::Int(vec![init_elem; *array_len])))
            }
        };

        self.all_inst.insert(Rc::as_ptr(&mem) as usize, mem.clone());

        let scope = self.top_scope();

        if scope.contains_key(&*name) {
            panic!("Variable {} already declared", name);
        }

        match var_type {
            Type::Array(_, _) => scope.insert(name.clone(), LValue { mem, idx: None }),
            _ => scope.insert(name.clone(), LValue { mem, idx: Some(0) }),
        };
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
        LValue { mem: self.all_inst.get_mut(&hash).unwrap().clone(), idx: Some(idx) }
    }
}

pub struct Interpreter {
    variables: InterpreterMemory,
    translation_unit: Rc<TranslationUnit>,
}


impl Interpreter {
    pub fn new(unit: TranslationUnit) -> Interpreter {
        Interpreter { variables: InterpreterMemory::new(), translation_unit: Rc::new(unit) }
    }

    pub fn empty() -> Interpreter {
        Interpreter { variables: InterpreterMemory::new(), translation_unit: Rc::new(TranslationUnit::new()) }
    }

    // integer types will be promoted to bigger type
    // (integer + ptr) or (ptr + integer) will be promoted to (ptr, int), swapping lhs/rhs if needed
    fn promote_binary_operands(&mut self, lhs: RValue, rhs: RValue) -> (RValue, RValue) {
        let target_type = rhs.get_type();

        match lhs {
            RValue::Int(_) => {
                match target_type {
                    Type::Int => return (lhs.clone(), rhs.clone()),
                    Type::Char => return (lhs.clone(), rhs.promote_to(Type::Int)),
                    Type::Ptr(_) => return (rhs.clone(), lhs.clone()),
                    _ => {}
                }
            }
            RValue::Char(_) => {
                match target_type {
                    Type::Int => return (lhs.promote_to(Type::Int), rhs.clone()),
                    Type::Char => return (lhs.clone(), rhs.clone()),
                    Type::Ptr(_) => return (rhs.promote_to(Type::Int), lhs.clone()),
                    _ => {}
                }
            }
            RValue::Void => {}
            RValue::Ptr(_, _, _) => {
                match target_type {
                    Type::Int => return (lhs.clone(), rhs.clone()),
                    Type::Char => return (lhs.clone(), rhs.promote_to(Type::Int)),
                    _ => {}
                }
            },
            _ => {}
        };

        panic!("Cannot promote values for binary op of types {} {}", lhs.get_type(), rhs.get_type());
    }

    fn op_add(&mut self, lhs: RValue, rhs: RValue) -> RValue {
        match lhs {
            // integer add
            RValue::Int(lhs) => RValue::Int(lhs + enum_cast!(rhs, RValue::Int)),
            RValue::Char(lhs) => RValue::Char(char::try_from(lhs as u32 + enum_cast!(rhs, RValue::Char) as u32).unwrap()),
            // pointer add
            RValue::Ptr(elem_type, hash, index) =>
                RValue::Ptr(elem_type, hash, (index as i32 + enum_cast!(rhs, RValue::Int)) as usize),
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_sub(&mut self, lhs: RValue, rhs: RValue) -> RValue {
        match lhs {
            RValue::Int(lhs) => RValue::Int(lhs - enum_cast!(rhs, RValue::Int)),
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_mul(&mut self, lhs: RValue, rhs: RValue) -> RValue {
        match lhs {
            RValue::Int(lhs) => RValue::Int(lhs * enum_cast!(rhs, RValue::Int)),
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_div(&mut self, lhs: RValue, rhs: RValue) -> RValue {
        match lhs {
            RValue::Int(lhs) => RValue::Int(lhs / enum_cast!(rhs, RValue::Int)),
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_neq(&mut self, lhs: RValue, rhs: RValue) -> RValue {
        match lhs {
            RValue::Int(lhs) => RValue::Int(if lhs != enum_cast!(rhs, RValue::Int) { 1 } else { 0 }),
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn op_gt(&mut self, lhs: RValue, rhs: RValue) -> RValue {
        match lhs {
            RValue::Int(lhs) => RValue::Int(if lhs > enum_cast!(rhs, RValue::Int) { 1 } else { 0 }),
            _ => panic!("Cannot apply add on {:?}, {:?}", lhs, rhs)
        }
    }

    fn print_value(&self, val: RValue) {
        match val {
            RValue::Int(val) => print!("{}", val),
            RValue::Char(val) => print!("{}", val),
            RValue::Void => panic!("Cannot print void value"),
            RValue::Ptr(target_type, hash, idx) => print!("{}*({:#x}:{})", target_type, hash, idx),
            RValue::Str(string) => print!("{}", string),
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
                    RValue::Ptr(expected_inner_type, hash, idx) => {
                        let value = self.variables.lookup_ptr(hash, idx);
                        let value_type = value.get_type();
                        assert_eq!(value_type, expected_inner_type,
                                   "Trying to dereference a pointer of type {}* pointing to data of element type {}.",
                                   expected_inner_type, value_type);
                        Ok(Value::LValue(value))
                    }
                    _ => panic!("Trying to dereference an expression that is no a pointer"),
                }
            }
            Operator::And => {
                let value = enum_cast!(self.visit_expr(expr)?, Value::LValue);
                return Ok(Value::RValue(RValue::Ptr(value.get_type(), Rc::as_ptr(&value.mem) as usize, value.idx.unwrap())));
            }
            _ => panic!("Unknown unary operator {:?}", op)
        }
    }

    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> Result<Value, Value> {
        let lhs = self.visit_expr(lhs)?;
        let rhs = self.visit_expr(rhs)?;

        if *op == Operator::Eq {
            let mut lhs = enum_cast!(lhs, Value::LValue);
            lhs.assign(rhs.to_rvalue());
            return Ok(Value::LValue(lhs));
        }

        let (lhs, rhs) = self.promote_binary_operands(lhs.to_rvalue(), rhs.to_rvalue());

        Ok(Value::RValue(match op {
            Operator::Plus => self.op_add(lhs, rhs),
            Operator::Minus => self.op_sub(lhs, rhs),
            Operator::Times => self.op_mul(lhs, rhs),
            Operator::Div => self.op_div(lhs, rhs),
            Operator::Neq => self.op_neq(lhs, rhs),
            Operator::Greater => self.op_gt(lhs, rhs),
            _ => panic!("Unsupported op {:?}", *op),
        }))
    }

    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> Result<Value, Value> {
        // TODO: make a table of default int values per type instead
        let value = match t {
            Type::Int => opt_init.as_ref().map_or(RValue::Int(0),
                                                  |expr| self.visit_expr(&expr).unwrap().to_rvalue()),
            Type::Char => opt_init.as_ref().map_or(RValue::Char('\0'),
                                                   |expr| self.visit_expr(&expr).unwrap().to_rvalue()),
            Type::Void => panic!("Cannot declare variable {} with type void.", name),
            Type::Ptr(t_var) => opt_init.as_ref().map_or(RValue::Ptr(t_var.deref().clone(), 0, 0),
                                                         |expr| { self.visit_expr(&expr).unwrap().to_rvalue() }),

            Type::Array(_, _) => opt_init.as_ref().map_or(RValue::Int(0),
                                                          |expr| self.visit_expr(&expr).unwrap().to_rvalue()),
        };

        self.variables.add_variable(name.clone(), t, value.clone());
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
                self.print_value(value.to_rvalue());
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
            Err(r) => Ok(Value::RValue(r.to_rvalue().promote_to(func.ret_type.clone())))
        }
    }

    fn visit_compound_statement(&mut self, compound: &CompoundStmt, variables: &HashMap<String, Value>) -> Result<Value, Value> {
        self.variables.push_scope();

        for (name, value) in variables {
            let r_value = value.to_rvalue();
            self.variables.add_variable(name.clone(), &r_value.get_type(), r_value);
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

