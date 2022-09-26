extern crate regex;
extern crate core;

mod ast;
mod interpreter;
mod parser;

use ast::Visitor;
use crate::interpreter::Interpreter;

fn main() {
    let tokens = parser::tokenize(r"
    int main(int a, int b)
    {
        while (a != b) {
            if (a > b) {
                a = a -b;
            } else {
                b = b - a;
            }
        }
        a;
    }
    ");

    // println!("{:?}", tokens);
    let trans_unit = parser::parse_translation_unit(tokens);


    let mut interp = Interpreter::new();
    let value = interp.visit_function_call(&trans_unit, "main", &vec![1071, 462]);
    println!("Result: {}", value);
}
