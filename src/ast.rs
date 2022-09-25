#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Operator {
    Plus,
    Minus,
    Times,
    Div,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Literal {
    pub value: i32,
}

#[derive(PartialEq, Debug)]
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

