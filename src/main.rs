extern crate regex;
extern crate core;

mod ast;
mod interpreter;
mod parser;

use ast::Visitor;
use crate::interpreter::Interpreter;

fn main() {
    let tokens = parser::tokenize(r#"

    int gcd(int a, int b) {
        while (a != b) {
            if (a > b) {
                a = a -b;
            } else {
                b = b - a;
            }
        }

        print(a);
        return a;
    }

    int fac(int i) {
        if(i > 1) {
          return i * fac(i-1);
        }

        return 1;
    }

    void foo() {
        if (5 > 3)  {
            return;
        }
    }

    void bar(int* p) {
        *p = *p + 1;
    }


    int main()
    {
        char c = 50;


        foo();

        int i = 1;
        int* p = &i;

        int** pp = &p;

        p[0] = 2;
        print(i);
        bar(&i);
        print(i);
        bar(p);
        print(i);

        **pp = 123;
        print(i);

       gcd(1071, 462);

       int[5] arr = 3;

       arr[0] = 5;
       bar(arr+2);
       print(arr[0]);
       print(arr[1]);
       print(arr[2]);

       print(c);

       int ci;
       ci = 5 + c;

       print(ci);

        print("arr+1");

       print(arr+1);

       return fac(5);
    }
    "#);

    // println!("{:?}", tokens);
    let trans_unit = parser::parse_translation_unit(tokens);


    let mut interp = Interpreter::new(trans_unit);
    let value = interp.visit_function_call("main", &vec![]).unwrap().to_rvalue();
    println!("Result: {:?}", value);
}
