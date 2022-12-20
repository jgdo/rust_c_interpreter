use std::collections::HashMap;
use std::env::var;
use regex::Regex;
use crate::ast;
use ast::Operator;
use crate::ast::{CompoundStmt, Expr, ExtDecl, FuncDef, Param, Stmt, TranslationUnit, Type, Variable, Literal};
use crate::ast::Operator::{Div, Eq, Greater, Minus, Neq, Plus, Times};


#[derive(Clone, PartialEq, Debug)]
pub enum Token {
    Literal(Literal),
    Operator(Operator),
    LP,
    RP,
    StopTag,
    Type(ast::Type),
    Sem,
    Identifier(String),
    While,
    Begin,
    // {
    End,
    // }
    If,
    Else,
    Comma,
    Return,
    BBL,
    // [
    BBR, // ]
}

struct Parser {
    tokens: Vec<Token>,
    current_pos: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        if self.current_pos >= self.tokens.len() {
            return &Token::StopTag;
        }

        return &self.tokens[self.current_pos];
    }

    fn take(&mut self) -> &Token {
        if self.current_pos >= self.tokens.len() {
            panic!("Cannot take token past all tokens.")
        }

        let res = &self.tokens[self.current_pos];
        self.current_pos += 1;
        return res;
    }

    fn parse_identifier(&mut self) -> String {
        let tok = self.take();

        match tok {
            Token::Identifier(ref name) => name.clone(),
            _ => panic!("Expected identifier"),
        }
    }

    fn accept(&mut self, expected: Token) {
        let actual = self.take();
        if expected != *actual {
            panic!("Expected {:?}, found {:?}", expected, actual)
        }
    }


    fn parse_primary_expr(&mut self) -> Expr {
        match self.take() {
            Token::Literal(ref e) => Expr::Literal(e.clone()),
            Token::LP => {
                let expr = self.parse_expr();
                self.accept(Token::RP);
                return expr;
            }
            Token::Identifier(ref name) => {
                let name = name.clone();
                match *self.peek() {
                    Token::LP => {
                        self.take();

                        let mut args: Vec<Expr> = vec![];
                        while *self.peek() != Token::RP {
                            args.push(self.parse_expr());
                            if *self.peek() == Token::Comma {
                                self.take();
                            }
                        }

                        self.accept(Token::RP);

                        Expr::FuncCall(name, args)
                    }
                    _ => Expr::Variable(Variable { name })
                }
            }
            Token::Operator(op) => {
                let op = *op;
                Expr::UnaryExpr(Box::new(self.parse_postfix_expr()), op)
            }
            _ => panic!("Unexpected Token")
        }
    }

    fn parse_postfix_expr(&mut self) -> Expr {
        let mut expr = self.parse_primary_expr();

        loop {
            match self.peek() {
                Token::BBL => {
                    self.take();
                    let index = self.parse_expr();
                    self.accept(Token::BBR);
                    expr = Expr::IndexExpr(Box::new(expr), Box::new(index))
                }
                _ => { break; }
            }
        }

        return expr;
    }

    fn parse_mul_expr(&mut self) -> Expr {
        let mut expr = self.parse_postfix_expr();

        loop {
            match self.peek() {
                Token::Operator(ref op) => {
                    if *op == Operator::Times || *op == Operator::Div {
                        let op = *op;
                        self.take();
                        let rhs = self.parse_postfix_expr();
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

    fn parse_add_expr(&mut self) -> Expr {
        let mut expr = self.parse_mul_expr();

        loop {
            match self.peek() {
                Token::Operator(ref op) => {
                    if *op == Operator::Plus || *op == Operator::Minus {
                        let op = *op;
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

    fn parse_expr(&mut self) -> Expr {
        let mut expr = self.parse_add_expr();

        loop {
            match self.peek() {
                Token::Operator(ref op) => {
                    if *op == Eq || *op == Neq || *op == Greater {
                        let op = *op;
                        self.take();
                        let rhs = self.parse_add_expr();
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

    fn parse_condition(&mut self) -> Expr {
        self.accept(Token::LP);
        let condition = self.parse_expr();
        self.accept(Token::RP);
        return condition;
    }

    fn parse_type(&mut self) -> Type {
        let mut var_type = match self.take() {
            Token::Type(t) => t.clone(),
            _ => panic!("Type expected"),
        };

        while *self.peek() == Token::Operator(Times){
            self.take();

            var_type = Type::Ptr(Box::new(var_type));
        }

        return var_type;
    }

    fn parse_statement(&mut self) -> Stmt {
        match self.peek() {
            Token::Type(_) => {
                let t = self.parse_type();
                let name = self.parse_identifier();

                let ret = if *self.peek() == Token::Operator(Operator::Eq) {
                    self.take();
                    let expr = self.parse_expr();
                    Stmt::VarDecl(t, name, Some(expr))
                } else {
                    Stmt::VarDecl(t, name, None)
                };

                self.accept(Token::Sem);
                return ret;
            }
            Token::While => {
                self.take();
                let condition = self.parse_condition();
                let body = self.parse_statement();
                let ret = Stmt::While(condition, Box::new(body));
                return ret;
            }
            Token::If => {
                self.take();
                let condition = self.parse_condition();
                let body_if = self.parse_statement();

                let body_else = if *self.peek() == Token::Else {
                    self.take();
                    Some(Box::new(self.parse_statement()))
                } else { None };

                let ret = Stmt::If(condition, Box::new(body_if), body_else);
                return ret;
            }
            Token::Begin => Stmt::Compound(Box::new(self.parse_compound_statement())),
            Token::Sem => {
                self.take();
                return Stmt::Empty;
            }
            Token::Return => {
                self.take();

                if *self.peek() == Token::Sem {
                    self.take();
                    return Stmt::Return(Expr::Literal(Literal::Void));
                }

                return Stmt::Return(self.parse_expr());
            }
            _ => {
                let ret = Stmt::Expr(self.parse_expr());
                self.accept(Token::Sem);
                return ret;
            }
        }
    }

    fn parse_compound_statement(&mut self) -> CompoundStmt {
        let mut statements: Vec<Stmt> = vec![];

        self.accept(Token::Begin);
        while *self.peek() != Token::End {
            statements.push(self.parse_statement());
        }
        self.accept(Token::End);

        return CompoundStmt { statements };
    }

    fn parse_func_def(&mut self) -> FuncDef {
        let ret_type = self.parse_type();
        let name = self.parse_identifier();
        self.accept(Token::LP);

        let mut params: Vec<Param> = vec![];
        while *self.peek() != Token::RP {
            let ptype = self.parse_type();
            let name = self.parse_identifier();
            params.push(Param { ptype, name });

            if *self.peek() == Token::Comma {
                self.take();
            }
        }
        self.accept(Token::RP);

        let body = self.parse_compound_statement();
        return FuncDef { ret_type, name, params, body };
    }

    fn parse_ext_decl(&mut self) -> ExtDecl {
        return ExtDecl::FuncDef(self.parse_func_def());
    }

    fn parse_translation_unit(&mut self) -> TranslationUnit {
        let mut functions: HashMap<String, FuncDef> = HashMap::new();

        while *self.peek() != Token::StopTag {
            match self.parse_ext_decl() {
                ExtDecl::FuncDef(f) => {
                    if functions.contains_key(&f.name) {
                        panic!("Function {} already declared", f.name);
                    }

                    functions.insert(f.name.clone(), f);
                }
            };
        }

        return TranslationUnit { functions };
    }
}

pub fn parse_expr(all_tokens: Vec<Token>) -> Expr {
    let mut parser = Parser { tokens: all_tokens, current_pos: 0 };

    let res = parser.parse_expr();
    if *parser.peek() != Token::StopTag {
        panic!("Unexpected tokens remaining");
    }
    return res;
}

pub fn parse_translation_unit(all_tokens: Vec<Token>) -> TranslationUnit {
    let mut parser = Parser { tokens: all_tokens, current_pos: 0 };

    let res = parser.parse_translation_unit();
    if *parser.peek() != Token::StopTag {
        panic!("Unexpected tokens remaining");
    }
    return res;
}

fn extract_token(str: &str) -> Token
{
    return match str.chars().next().unwrap() {
        'A'..='Z' | 'a'..='z' | '_' => {
            match str {
                // TODO: clean up with keywords table
                "int" => Token::Type(Type::Int),
                "void" => Token::Type(Type::Void),
                "while" => Token::While,
                "if" => Token::If,
                "else" => Token::Else,
                "return" => Token::Return,
                _ => Token::Identifier(str.parse().unwrap()),
            }
        }
        '0'..='9' => Token::Literal(Literal::Int(str.parse::<i32>().unwrap())),
        '(' => Token::LP,
        ')' => Token::RP,
        '{' => Token::Begin,
        '}' => Token::End,
        '+' => Token::Operator(Plus),
        '-' => Token::Operator(Minus),
        '*' => Token::Operator(Times),
        '/' => Token::Operator(Div),
        '=' => Token::Operator(Eq),
        '!' => Token::Operator(Neq), // TODO will match more
        '>' => Token::Operator(Greater),
        ';' => Token::Sem,
        ',' => Token::Comma,
        '&' => Token::Operator(Operator::And),
        '[' => Token::BBL,
        ']' => Token::BBR,
        _ => panic!("cannot tokenize '{}'", str),
    };
}


pub fn tokenize(text: &str) -> Vec<Token> {
    // TODO this will actually match more than it should, fix eventually
    let re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)|\+|-|\*|/|(=)|(!=)|([0-9]+)|\(|\)|;|\{|}|>|,|&|\[|]").unwrap();

    let mut res: Vec<Token> = vec![];
    for m in re.find_iter(text) {
        res.push(extract_token(m.as_str()));
    }

    return res;
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use ast::Expr;
    use crate::ast::{Literal, Variable};

    use crate::parser as tok;

    #[test]
    fn tokenize() {
        let tokes = tok::tokenize("(5 +3)- (2 )     *   7");

        let expected = vec![
            tok::Token::LP,
            tok::Token::Literal(Literal::Int(5)),
            tok::Token::Operator(ast::Operator::Plus),
            tok::Token::Literal(Literal::Int(3)),
            tok::Token::RP,
            tok::Token::Operator(ast::Operator::Minus),
            tok::Token::LP,
            tok::Token::Literal(Literal::Int(2)),
            tok::Token::RP,
            tok::Token::Operator(ast::Operator::Times),
            tok::Token::Literal(Literal::Int(7)),
        ];

        assert_eq!(tokes, expected);
    }

    #[test]
    fn parse_expr() {
        let tokens = vec![
            tok::Token::Identifier("a".to_string()),
            tok::Token::Operator(ast::Operator::Eq),
            tok::Token::LP,
            tok::Token::Literal(Literal::Int(5)),
            tok::Token::Operator(ast::Operator::Plus),
            tok::Token::Literal(Literal::Int(3)),
            tok::Token::RP,
            tok::Token::Operator(ast::Operator::Minus),
            tok::Token::LP,
            tok::Token::Literal(Literal::Int(2)),
            tok::Token::RP,
            tok::Token::Operator(ast::Operator::Times),
            tok::Token::Literal(Literal::Int(7)),
        ];

        let expr = tok::parse_expr(tokens);

        let expected =
            Expr::BinaryExpr(
                Box::new(Expr::Variable(Variable { name: "a".to_string() })),
                Box::new(Expr::BinaryExpr(
                    Box::new(
                        Expr::BinaryExpr(
                            Box::new(Expr::Literal(Literal::Int(5))),
                            Box::new(Expr::Literal(Literal::Int(2))),
                            ast::Operator::Plus,
                        )),
                    Box::new(Expr::BinaryExpr(
                        Box::new(Expr::Literal(Literal::Int(2))),
                        Box::new(Expr::Literal(Literal::Int(7))),
                        ast::Operator::Times,
                    )),
                    ast::Operator::Minus,
                )),
                ast::Operator::Eq,
            );

        assert_eq!(expr, expected)
    }
}
