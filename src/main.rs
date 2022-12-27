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

    void swap(int* lhs, int* rhs) {
        int tmp = *lhs;
        *lhs = *rhs;
        *rhs = tmp;
    }

    void sort(int* arr, int len) {
        for(int i = 0; len > i; i = i+1) {
           for(int j = i+1; len > j; j = j+1) {
                if(arr[i] > arr[j]) {
                    swap(&arr[i], arr+j);
                }
           }
        }
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

       int[5] arr;
       arr[0] = 23;
       arr[1] = 5;
       arr[2] = 42;
       arr[3] =6;
       arr[4] = 1;

        sort(arr, 5);
        print("sorted arr:");
        for(int j = 0; 5 > j; j = j+1) {
          print(arr[j]);
        }

       return fac(5);
    }
    "#);

    // println!("{:?}", tokens);
    let trans_unit = parser::parse_translation_unit(tokens);


    let mut interp = Interpreter::new(trans_unit);
    let value = interp.visit_function_call("main", &vec![]).unwrap().to_rvalue();
    println!("Result: {:?}", value);
}
