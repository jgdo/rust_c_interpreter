extern crate regex;

mod ast;
mod interpreter;
mod parser;

use ast::Visitor;
use crate::interpreter::Interpreter;

fn main() {
    let tokens = parser::tokenize(r"
    int a = 1071;
    int b = 462;
    while (a != b) {
        if (a > b) {
            a = a -b;
        } else {
            b = b - a;
        }
    }
    a;
    ");
    let statements = parser::parse_compound_stmt(tokens);

    let mut interp = Interpreter::new();
    let value = interp.visit_compound_statement(&statements);
    println!("Result: {}", value);
}
