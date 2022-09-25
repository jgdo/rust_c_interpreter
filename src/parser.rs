use crate::ast;
use ast::Operator;
use crate::ast::Expr;

#[derive(Copy, Clone, PartialEq)]
pub struct Literal {
    pub value: i32,
}

#[derive(Copy, Clone, PartialEq)]
pub enum Token {
    Literal(Literal),
    Operator(Operator),
    LP,
    RP,
    End,
}

struct Parser {
    tokens: Vec<Token>,
    current_pos: usize,
}

impl Parser {
    fn peek(&self) -> Token {
        if self.current_pos >= self.tokens.len() {
            return Token::End;
        }

        return self.tokens[self.current_pos];
    }

    fn take(&mut self) -> Token {
        if self.current_pos >= self.tokens.len() {
            panic!("Cannot take token past all tokens.")
        }

        let res = self.tokens[self.current_pos];
        self.current_pos += 1;
        return res;
    }

    fn accept(&mut self, expected: Token) {
        let actual = self.take();
        if expected != actual {
            panic!("Expected token did not match actual token")
        }
    }

    fn parse_primary_expr(&mut self) -> ast::Expr {
        match self.peek() {
            Token::Literal(ref e) => {
                self.take();
                let expr = ast::Expr::Literal(ast::Literal { value: e.value });
                return expr;
            }
            Token::Operator(_) => { panic!("Unexpected operator") }
            Token::LP => {
                self.take();
                let expr = self.parse_expr();
                self.accept(Token::RP);
                return expr;
            }
            Token::RP => { panic!("Unexpected ')'") }
            Token::End => { panic!("Expression to parse is empty") }
        }
    }

    fn parse_mul_expr(&mut self) -> ast::Expr {
        let mut expr = self.parse_primary_expr();

        loop {
            match self.peek() {
                Token::Operator(op) => {
                    if op == Operator::Times || op == Operator::Div {
                        self.take();
                        let rhs = self.parse_primary_expr();
                        expr = Expr::BinaryExpr(Box::new(expr), Box::new(rhs), op);
                    } else {
                        break;
                    }
                }
                _ => { break; }
            }
        }

        return expr;
    }

    fn parse_expr(&mut self) -> ast::Expr {
        let mut expr = self.parse_mul_expr();

        loop {
            match self.peek() {
                Token::Operator(op) => {
                    if op == Operator::Plus || op == Operator::Minus {
                        self.take();
                        let rhs = self.parse_mul_expr();
                        expr = Expr::BinaryExpr(Box::new(expr), Box::new(rhs), op);
                    } else {
                        break;
                    }
                }
                _ => { break; }
            }
        }

        return expr;
    }
}


pub fn parse_expr(all_tokens: Vec<Token>) -> ast::Expr {
    let mut parser = Parser { tokens: all_tokens, current_pos: 0 };

    let res = parser.parse_expr();
    if parser.peek() != Token::End {
        panic!("Unexpected tokens remaining");
    }
    return res;
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use ast::Expr;

    use crate::parser as tok;

    #[test]
    fn parse_expr() {
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
            tok::Token::Literal(tok::Literal { value: 7 }),
        ];

        let expr = tok::parse_expr(tokens);

        let expected = Expr::BinaryExpr(
            Box::new(
                Expr::BinaryExpr(
                    Box::new(Expr::Literal(ast::Literal { value: 5 })),
                    Box::new(Expr::Literal(ast::Literal { value: 3 })),
                    ast::Operator::Plus,
                )),
            Box::new(Expr::BinaryExpr(
                Box::new(Expr::Literal(ast::Literal { value: 2 })),
                Box::new(Expr::Literal(ast::Literal { value: 7 })),
                ast::Operator::Times,
            )),
            ast::Operator::Minus,
        );

        assert_eq!(expr, expected)
    }
}