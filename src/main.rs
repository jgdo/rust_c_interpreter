extern crate regex;
extern crate core;

mod ast;
mod interpreter;
mod parser;

use ast::Visitor;
use crate::interpreter::Interpreter;

fn main() {
    let tokens = parser::tokenize(r"

    int gcd(int a, int b) {
        while (a != b) {
            if (a > b) {
                a = a -b;
            } else {
                b = b - a;
            }
        }
        a;
    }

    int fac(int i) {
        if(i > 1) {
          i * fac(i-1);
        } else {
          1;
        }
    }

    int main()
    {
       gcd(1071, 462);
       fac(5);
    }
    ");

    // println!("{:?}", tokens);
    let trans_unit = parser::parse_translation_unit(tokens);


    let mut interp = Interpreter::new(trans_unit);
    let value = interp.visit_function_call("main", &vec![]);
    println!("Result: {}", value);
}
