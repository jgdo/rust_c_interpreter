extern crate regex;

mod ast;
mod interpreter;
mod parser;
use ast::Visitor;

fn main() {
    let tokens = parser::tokenize("(5+3)-(2)*7");
    let expr = parser::parse_expr(tokens);

    let interp = interpreter::Interpreter{};
    let result = interp.visit_expr(&expr);

    println!("{}", result);
}
