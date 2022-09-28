use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Operator {
    Plus,
    Minus,
    Times,
    Div,
    Eq,
    Neq,
    Greater,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Literal {
    pub value: i32,
}

#[derive(PartialEq, Debug)]
pub struct Variable {
    pub name: String,
}

#[derive(PartialEq, Debug)]
pub enum Expr {
    Variable(Variable),
    Literal(Literal),
    BinaryExpr(Box<Expr>, Box<Expr>, Operator),
    FuncCall(String, Vec<Expr>),
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Type {
    Int,
}

#[derive(PartialEq, Debug)]
pub enum Stmt {
    Expr(Expr),
    VarDecl(Type, String, Option<Expr>),
    Compound(Box<CompoundStmt>),
    While(Expr, Box<Stmt>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    Empty,
}

#[derive(PartialEq, Debug)]
pub struct CompoundStmt {
    pub statements: Vec<Stmt>,
}

#[derive(PartialEq, Debug)]
pub struct Param {
    pub ptype: Type,
    pub name: String,
}

#[derive(PartialEq, Debug)]
pub struct FuncDef {
    pub ret_type: Type,
    pub name: String,
    pub params: Vec<Param>,
    pub body: CompoundStmt,
}

#[derive(PartialEq, Debug)]
pub enum ExtDecl {
    FuncDef(FuncDef),
}

#[derive(PartialEq, Debug)]
pub struct TranslationUnit {
    pub functions: HashMap<String, FuncDef>,
}

impl TranslationUnit {
    pub fn new() -> TranslationUnit {
        TranslationUnit { functions: HashMap::new() }
    }
}

pub trait Visitor<R> {
    fn visit_literal(&mut self, e: &Literal) -> R;
    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> R;
    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> R;
    fn visit_var(&mut self, var: &Variable) -> R;
    fn visit_while(&mut self, cond: &Expr, body: &Stmt) -> R;
    fn visit_if(&mut self, cond: &Expr, body_if: &Stmt, opt_body_else: &Option<Box<Stmt>>) -> R;
    fn visit_empty(&mut self) -> R;
    fn visit_function_call(&mut self, name: &str, args: &Vec<R>) -> R;
    fn visit_compound_statement(&mut self, compound: &CompoundStmt, variables: &HashMap<String, i32>) -> R;

    fn visit_expr(&mut self, e: &Expr) -> R {
        match e {
            Expr::Literal(ref lit) => self.visit_literal(lit),
            Expr::BinaryExpr(ref lhs, ref rhs, ref op) => self.visit_binary_exp(lhs, rhs, op),
            Expr::Variable(ref var) => self.visit_var(var),
            Expr::FuncCall(ref name, ref args) => {
                let args_values = &args.iter().map(|e| self.visit_expr(e)).collect();
                self.visit_function_call(name, args_values)
            }
        }
    }

    fn visit_statement(&mut self, stmt: &Stmt) -> R {
        match stmt {
            Stmt::Expr(ref expr) => self.visit_expr(expr),
            Stmt::VarDecl(ref t, ref name, ref opt_init) => self.visit_var_decl(t, name, opt_init),
            Stmt::Compound(ref compound) => self.visit_compound_statement(compound, &HashMap::new()),
            Stmt::While(ref cond, ref body) => self.visit_while(cond, body),
            Stmt::If(cond, ref body_if, ref opt_body_else) => self.visit_if(cond, body_if, opt_body_else),
            Stmt::Empty => self.visit_empty()
        }
    }
}

