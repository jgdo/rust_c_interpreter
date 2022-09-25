extern crate regex;

mod ast;
mod interpreter;
mod parser;

use ast::Visitor;
use crate::interpreter::Interpreter;

fn main() {
    let tokens = parser::tokenize("int a = 5; a = a+2; a;");
    let statements = parser::parse_compound_stmt(tokens);

    let mut interp = Interpreter::new();
    let value = interp.visit_compound_statement(&statements);
    println!("Result: {}", value);
}
