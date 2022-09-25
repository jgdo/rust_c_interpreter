#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Operator {
    Plus,
    Minus,
    Times,
    Div,
    Eq,
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
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Type {
    Int,
}

#[derive(PartialEq, Debug)]
pub enum Stmt {
    Expr(Expr),
    VarDecl(Type, String, Option<Expr>),
}

#[derive(PartialEq, Debug)]
pub struct CompoundStmt {
    pub(crate) statements: Vec<Stmt>,
}

pub trait Visitor<R> {
    fn visit_literal(&mut self, e: &Literal) -> R;
    fn visit_binary_exp(&mut self, lhs: &Expr, rhs: &Expr, op: &Operator) -> R;
    fn visit_var_decl(&mut self, t: &Type, name: &String, opt_init: &Option<Expr>) -> R;
    fn visit_var(&mut self, var: &Variable) -> R;

    fn visit_expr(&mut self, e: &Expr) -> R {
        match e {
            Expr::Literal(ref lit) => self.visit_literal(lit),
            Expr::BinaryExpr(ref lhs, ref rhs, ref op) => self.visit_binary_exp(lhs, rhs, op),
            Expr::Variable(ref var) => self.visit_var(var),
        }
    }

    fn visit_statement(&mut self, stmt: &Stmt) -> R {
        match stmt {
            Stmt::Expr(ref expr) => self.visit_expr(expr),
            Stmt::VarDecl(ref t, ref name, ref opt_init) => self.visit_var_decl(t, name, opt_init),
        }
    }

    fn visit_compound_statement(&mut self, compound: &CompoundStmt) -> R {
        let (last, first) = compound.statements.split_last().unwrap();

        for stmt in first {
            self.visit_statement(stmt);
        }

        return self.visit_statement(last);
    }
}

