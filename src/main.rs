extern crate core;

mod ast;
mod interpreter;
mod parser;

use crate::ast::Visitor;
use crate::parser as tok;
use crate::tok::parse_expr;


fn main() {
    let interp = interpreter::Interpreter {};

    let tokens = vec![
        tok::Token::LP,
        tok::Token::Literal(tok::Literal { value: 5 }),
        tok::Token::Operator(ast::Operator::Plus),
        tok::Token::Literal(tok::Literal { value: 3 }),
        tok::Token::RP,
        tok::Token::Operator(ast::Operator::Minus),
        tok::Token::LP,
        tok::Token::Literal(tok::Literal { value: 2 }),
        tok::Token::RP,
        tok::Token::Operator(ast::Operator::Times),
        tok::Token::Literal(tok::Literal { value: 5 }),
    ];

    let expr = parse_expr(tokens);
    let value = interp.visit_expr(&expr);
    print!("{}", value);
}
