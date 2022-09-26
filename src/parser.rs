use std::collections::HashMap;
use std::os::linux::raw::stat;
use regex::Regex;
use crate::ast;
use ast::Operator;
use crate::ast::{CompoundStmt, Expr, ExtDecl, FuncDef, Param, Stmt, TranslationUnit, Type, Variable};
use crate::ast::Operator::{Div, Eq, Greater, Minus, Neq, Plus, Times};
use crate::ast::Stmt::Empty;

#[derive(Clone, PartialEq, Debug)]
pub struct Literal {
    pub value: i32,
}

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
            panic!("Expected token did not match actual token")
        }
    }

    fn parse_primary_expr(&mut self) -> Expr {
        match self.take() {
            Token::Literal(ref e) => Expr::Literal(ast::Literal { value: e.value }),
            Token::LP => {
                let expr = self.parse_expr();
                self.accept(Token::RP);
                return expr;
            }
            Token::Identifier(ref name) => Expr::Variable(Variable { name: name.clone() }),
            _ => panic!("Unexpected Token")
        }
    }

    fn parse_mul_expr(&mut self) -> Expr {
        let mut expr = self.parse_primary_expr();

        loop {
            match self.peek() {
                Token::Operator(ref op) => {
                    if *op == Operator::Times || *op == Operator::Div {
                        let op = *op;
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
        match self.take() {
            Token::Type(t) => *t,
            _ => panic!("Type expected"),
        }
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
                return Empty;
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
        }
        self.accept(Token::RP);

        let body = self.parse_compound_statement();
        return FuncDef { ret_type, name, params, body};
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

pub fn parse_compound_stmt(all_tokens: Vec<Token>) -> CompoundStmt {
    let mut parser = Parser { tokens: all_tokens, current_pos: 0 };

    let res = parser.parse_compound_statement();
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

pub fn tokenize(text: &str) -> Vec<Token> {
    // TODO this will actually match more than it should, fix eventually
    let re = Regex::new(r"([a-zA-Z_][a-zA-Z0-9_]*)|\+|-|\*|/|(=)|(!=)|([0-9]+)|\(|\)|;|\{|}|>").unwrap();

    let mut res: Vec<Token> = vec![];
    for m in re.find_iter(text) {
        let m_str = m.as_str();

        match m_str.chars().next().unwrap() {
            'A'..='Z' | 'a'..='z' | '_' => {
                match m_str {
                    "int" => res.push(Token::Type(Type::Int)),
                    "while" => res.push(Token::While),
                    "if" => res.push(Token::If),
                    "else" => res.push(Token::Else),
                    _ => res.push(Token::Identifier(m_str.parse().unwrap()))
                }
            }
            '0'..='9' => res.push(Token::Literal(Literal { value: m_str.parse::<i32>().unwrap() })),
            '(' => res.push(Token::LP),
            ')' => res.push(Token::RP),
            '{' => res.push(Token::Begin),
            '}' => res.push(Token::End),
            '+' => res.push(Token::Operator(Plus)),
            '-' => res.push(Token::Operator(Minus)),
            '*' => res.push(Token::Operator(Times)),
            '/' => res.push(Token::Operator(Div)),
            '=' => res.push(Token::Operator(Eq)),
            '!' => res.push(Token::Operator(Neq)), // TODO will match more
            '>' => res.push(Token::Operator(Greater)),
            ';' => res.push(Token::Sem),
            _ => { panic!("cannot tokenize '{}'", m_str) }
        }
    }

    return res;
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use ast::Expr;
    use crate::ast::Variable;

    use crate::parser as tok;

    #[test]
    fn tokenize() {
        let tokes = tok::tokenize("(5 +3)- (2 )     *   7");

        let expected = vec![
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

        assert_eq!(tokes, expected);
    }

    #[test]
    fn parse_expr() {
        let tokens = vec![
            tok::Token::Identifier("a".to_string()),
            tok::Token::Operator(ast::Operator::Eq),
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

        let expected =
            Expr::BinaryExpr(
                Box::new(Expr::Variable(Variable { name: "a".to_string() })),
                Box::new(Expr::BinaryExpr(
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
                )),
                ast::Operator::Eq,
            );

        assert_eq!(expr, expected)
    }
}
