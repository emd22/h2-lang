from enum import Enum

class TokenType(Enum):
    NONE = -1
    IDENTIFIER = 0,
    OP_PLUS = 1,
    OP_MINUS = 2,
    OP_MUL  = 3,
    OP_DIV   = 4
    LPAREN = 5,
    RPAREN = 6,
    LBRACE = 7,
    RBRACE = 8,
    NUMBER = 9,
    STRING = 10,
    SYMBOL = 11,
    COMMA = 12,
    SEMICOLON = 13,

class Token():
    def __init__(self, type):
        self.type = type
        self.index = 0
        self.data = ""
    def __repr__(self):
        return "Token[T:{}, {}]".format(self.type, self.data)

class Lexer():
    def __init__(self, buffer):
        self.splits = ".,+-*/=(){};"
        self.ws = " \t\n\0"
        self.buffer = buffer
        self.tokens = []

    def push_token(self, token):
        token.index = len(self.tokens)
        self.tokens.append(self.ttoken(token))

    def lex(self):
        token = Token(TokenType.IDENTIFIER)
        token.data = ""
        in_str = False
        for ch in self.buffer:
            if ch == '"':
                in_str = not in_str
            if in_str:
                token.data += ch
                continue
            if ch in self.splits or ch in self.ws:
                if token.data != "":
                    #print("TTOK: {}".format(token.data))
                    self.push_token(token)
                    token = Token(TokenType.IDENTIFIER)
                if ch in self.splits:
                    split = Token(TokenType.IDENTIFIER)
                    split.data = ch
                    self.push_token(split)
                continue
            else:
                token.data += ch

    def ttoken(self, token):
        data = token.data
        symbols = '%#'
        if data == '+':
            token.type = TokenType.OP_PLUS
        elif data == ',':
            token.type = TokenType.COMMA
        elif data == '-':
            token.type = TokenType.OP_MINUS
        elif data == '*':
            token.type = TokenType.OP_MUL
        elif data == '/':
            token.type = TokenType.OP_DIV
        elif data == '(':
            token.type = TokenType.LPAREN
        elif data == ')':
            token.type = TokenType.RPAREN
        elif data == '{':
            token.type = TokenType.LBRACE
        elif data == '}':
            token.type = TokenType.RBRACE
        elif data == ';':
            token.type = TokenType.SEMICOLON
        elif data[0] == '"' and data[-1] == '"':
            token.type = TokenType.STRING
        elif data.isnumeric():
            token.type = TokenType.NUMBER
        elif data in symbols:
            token.type = TokenType.SYMBOL
        else:
            token.type = TokenType.IDENTIFIER
        return token

    def print(self):
        for token in self.tokens:
            print("New token : '{}' :: {}".format(token.data, token.type))

class AstNode():
    def __init__(self, nodetype, children=None):
        self.type = nodetype
        self.children = children

class NodeBlock(AstNode):
    def __init__(self, statements):
        self.children = statements
        self.token = Token(TokenType.NONE)

class NodeUnaryOp(AstNode):
    def __init__(self, op, expr):
        self.token = op
        self.children = [expr]

class NodeBinOp(AstNode):
    def __init__(self, left, op, right):
        self.children = [left, right]
        self.token = op

class NodeLiteral(AstNode):
    def __init__(self, token):
        self.token = token
        self.children = []
        self.value = token.data

class NodeSymbol(AstNode):
    def __init__(self, token):
        self.token = token
        self.children = []

class NodeArgList(AstNode):
    def __init__(self, nodelist):
        self.children = nodelist
        self.token = Token(TokenType.NONE)
    def __repr__(self):
        arg_str = ", ".join(x.token.data for x in self.children)
        return "[{}]".format(arg_str)

class NodePCall(AstNode):
    def __init__(self, func, arglist):
        self.token = func.token
        self.children = [func, arglist]

class Parser():
    def __init__(self, tokens):
        self.tokens = tokens
        self.tokens.append(Token(TokenType.IDENTIFIER))
        self.token_index = 0
        self.token = self.tokens[0]

    def next_token(self):
        self.token_index += 1
        self.token = self.tokens[self.token_index]
        return self.token

    def peek_token(self, offset=1):
        if self.token_index+offset < 0 or self.token_index+offset >= len(self.tokens):
            return Token(TokenType.NONE)
        return self.tokens[self.token_index+offset]

    def eat(self, token_type):
        if self.token.type == token_type:
            self.next_token()
        else:
            raise Exception("Error eating token ({} != {})".format(self.token.type, token_type))

    def parse_arglist(self):
        args = []
        self.eat(TokenType.LPAREN)
        while self.token.type != TokenType.RPAREN:
            #print("TT: {}".format(self.token.data))
            args.append(self.parse_expr())
            if self.token.type != TokenType.COMMA:
                break
            self.eat(TokenType.COMMA)
        self.eat(TokenType.RPAREN)
        return NodeArgList(args)

    def parse_factor(self):
        token = self.token
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return NodeLiteral(token)

        elif token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            if (self.token.type == TokenType.LPAREN):
                symbol = NodeSymbol(token)
                args = self.parse_arglist()
                return NodePCall(symbol, args)

            return NodeSymbol(token)

        elif token.type == TokenType.STRING:
            self.eat(TokenType.STRING)
            return NodeLiteral(token)

        elif token.type == TokenType.LPAREN:
            #if self.peek_token(-1).type == TokenType.IDENTIFIER:
            #    symbol = NodeSymbol(self.peek_token(-1))
            #    args = self.parse_arglist()
            #    return NodePCall(symbol, args)
            self.eat(TokenType.LPAREN)
            node = self.parse_expr()
            self.eat(TokenType.RPAREN)
            return node

        elif token.type == TokenType.LBRACE:
            #self.eat(TokenType.LBRACE)
            node = self.parse_block()
            #self.eat(TokenType.RBRACE)
            return node

        elif token.type == TokenType.OP_PLUS:
            self.eat(TokenType.OP_PLUS)
            return NodeUnaryOp(token, self.parse_factor())
        elif token.type == TokenType.OP_MINUS:
            self.eat(TokenType.OP_MINUS)
            return NodeUnaryOp(token, self.parse_factor())
        print("parse_factor: unknown token type {}".format(token.type))

    def parse_term(self):
        node = self.parse_factor()
        while True:
            token = self.token
            if token.type == TokenType.OP_MUL:
                self.eat(TokenType.OP_MUL)
            elif token.type == TokenType.OP_DIV:
                self.eat(TokenType.OP_DIV)
            else:
                break
            node = NodeBinOp(node, token, self.parse_factor())
        return node

    def parse_expr(self):
        node = self.parse_term()
        while True:
            token = self.token
            if token.type == TokenType.OP_PLUS:
                self.eat(TokenType.OP_PLUS)
            elif token.type == TokenType.OP_MINUS:
                self.eat(TokenType.OP_MINUS)
            else:
                break

            node = NodeBinOp(node, token, self.parse_term())
        return node

    def parse_statement(self):
        node = self.parse_expr()
        self.eat(TokenType.SEMICOLON)
        return node

    def parse_block(self):
        self.eat(TokenType.LBRACE)
        statements = []
        while self.token.type != TokenType.RBRACE:
            statements.append(self.parse_statement())
        self.eat(TokenType.RBRACE)
        block = NodeBlock(statements)
        return block

    def parse(self, node):
        statements = []
        while True:
            statements.append(self.parse_statement())
            if self.peek_token().type == TokenType.NONE:
                break
        block = NodeBlock(statements)
        return block

    def print_tree(self, node, depth=0, index=0):
        print('\t'*depth, end='')
        print("{}. [{} | children:{}] => {}".format(index, type(node).__name__, len(node.children), node.token.data))
        ind = 0
        for child in node.children:
            self.print_tree(child, depth+1, ind)
            ind += 1

class NodeVisitor(object):
    def visit(self, node):
        method_name = "visit_"+type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    def generic_visit(self, node):
        raise Exception("No visit method found for {}".format(type(node).__name__))

class PPDefinition():
    def __init__(self, name, data, ifunc=None, fnode=None):
        self.name = name
        self.data = data
        self.func = ifunc
        self.fnode = fnode
    def __repr__(self):
        return "[PPDef: {} | {}]".format(self.name, self.data)
        
class CodeGeneration():
    def __init__(self):
        self.text = ""
        self.data = ""
        self.bss = ""
        
        self.pre = ""
    def output(self):
        print("=== ASM ===")
        output = \
        """
{}
section .text
{}
section .data
{}
section .bss
{}
        """.format(self.pre, self.text, self.data, self.bss)
        return output

class PreProcess(NodeVisitor):
    def __init__(self, tokens, ast):
        self.tokens = tokens
        self.ast = ast
        self.ostack = []
        self.cg = CodeGeneration()
        self.definitions = [
            PPDefinition("_macro", "<INTERNAL>", self.int_define),
            PPDefinition("_fmacro", "<INTERNAL>", self.int_fdefine),
            PPDefinition("_delete", "<INTERNAL>", self.int_delete),
            PPDefinition("_setmacro", "<INTERNAL>", self.int_dset),
            PPDefinition("_debugvar", "<INTERNAL>", self.int_dprt),
            PPDefinition("_add", "<INTERNAL>", self.int_add),
            PPDefinition("_T", "<INTERNAL>", self.int_gettok),
            # use the builtin variable '$' to get current token index
            PPDefinition("_dvars", "<INTERNAL>", self.int_dvars),
            
            PPDefinition("_push", "", self.int_push),
            PPDefinition("_pop", "", self.int_pop),
            
            # code generation
            PPDefinition("_appsec", "<INTERNAL>", self.int_appsec),
        ]

    def get_def(self, name):
        pcall = None
        for definition in self.definitions:
            if definition.name == name:
                return definition
                
    def set_def(self, name, data):
        pcall = None
        for definition in self.definitions:
            if definition.name == name:
                definition.data = data
                return definition
                
    def int_appsec(self, fcall_node):
        arg_node = fcall_node.children[1]
        section = int(self.visit(arg_node.children[0]))
        output = None
        
        for child in arg_node.children[1:]:
            value = str(self.visit(child))
            if (section == 0):
                self.cg.text += value
            elif (section == 1):
                self.cg.data += value
            elif (section == 2):
                self.cg.bss += value
            elif (section == 3):
                self.cg.pre += value
                
    def int_asm(self, fcall_node):
        arg_node = fcall_node.children[1]
        for child in arg_node.children:
            self.asm_out += self.visit(child)
        return 0
        
    def int_push(self, fcall_node):
        arg_node = fcall_node.children[1]
        for child in arg_node.children:
            self.ostack.append(self.visit(child))

    def int_pop(self, fcall_node):
        return self.visit(self.ostack.pop())

    def int_gettok(self, fcall_node):
        arg_node = fcall_node.children[1]

        index = int(self.visit(arg_node.children[0]))
        print("Gettok index {}".format(index))
        return self.tokens[index].data

    def int_dvars(self, fcall_node):
        for macro in self.definitions:
            print(macro)

    def int_dprt(self, fcall_node):
        arg_node = fcall_node.children[1]

        name = arg_node.children[0].token.data
        pcall = self.get_def(name)
        value = self.visit(pcall.data)
        print("Variable '{}' value is '{}'".format(name, value))

    def int_add(self, fcall_node):
        arg_node = fcall_node.children[1]
        arg0 = self.visit(arg_node.children[0])
        arg1 = self.visit(arg_node.children[1])
        return int(arg0)+int(arg1)

    def int_dset(self, fcall_node):
        arg_node = fcall_node.children[1]
        name = arg_node.children[0].token.data
        data = arg_node.children[1]
        for macro in self.definitions:
            if macro.name == name:
                macro.data = data
                break
        return data

    def int_fdefine(self, fcall_node):
        arg_node = fcall_node.children[1]
        name = arg_node.children[0].token.data
        data = arg_node.children[1]
        self.definitions.append(PPDefinition(name, data, fnode=data))
        return data
        
    def int_delete(self, fcall_node):
        arg_node = fcall_node.children[1]
        name = arg_node.children[0].token.data
        for definition in self.definitions:
            if definition.name == name:
                self.definitions.delete(definition)

    def int_define(self, fcall_node):
        arg_node = fcall_node.children[1]
        name = arg_node.children[0].token.data
        data = arg_node.children[1]
        self.definitions.append(PPDefinition(name, data))
        return data

    def visit_NodeUnaryOp(self, node):
        if node.token.type == TokenType.OP_PLUS:
            return +self.visit(node.children[0])
        elif node.token.type == TokenType.OP_MINUS:
            return -self.visit(node.children[0])

    def visit_NodeBinOp(self, node):
        arg0 = self.visit(node.children[0])
        arg1 = self.visit(node.children[1])
        if node.token.type == TokenType.OP_PLUS:
            return arg0+arg1
        elif node.token.type == TokenType.OP_MINUS:
            return arg0-arg1
        elif node.token.type == TokenType.OP_MUL:
            return arg0*arg1
        elif node.token.type == TokenType.OP_DIV:
            return arg0/arg1

    def visit_NodeBlock(self, node):
        for child in node.children:
            self.visit(child)

    def visit_NodeLiteral(self, node):
        if node.token.data.isnumeric():
            return int(node.token.data)
        if node.token.data[0] == '"' and node.token.data[-1] == '"':
            return node.token.data[1:-1]
        return node.token.data

    def visit_NodeSymbol(self, node):
        # get current token index
        if node.token.data == '$':
            print("return index of {}".format(node.token.index))
            return node.token.index

        if node.token.data[0] == '%':
            macro = self.get_def(node.token.data)
            return self.visit(macro.data)
        return node.token.data
        #val = self.get_def(node.token.data)
        #print("SYMBDAT :: {}".format(self.definitions))
        #return val.data

    def visit_NodeArgList(self, node):
        args = []
        for child in node.children:
            args.append(self.visit(child))
        return args

    def visit_NodePCall(self, node):
        symbol = node.children[0]
        pcall = self.get_def(self.visit(symbol))
        if pcall.fnode != None:
            args = node.children[1].children
            for arg in reversed(args):
                self.ostack.append(arg)
            rval = self.visit(pcall.fnode)
        elif pcall.func != None:
            rval = pcall.func(node)

        print("Call '{}' with args {}".format(pcall.name, node.children[1]))
        return rval

    def pp(self):
        self.visit(self.ast)

def main():
    input = """
        {
            _macro(%_TEXT, 0);
            _macro(%_DATA, 1);
            _macro(%_BSS, 2);
            _macro(%_PRE, 3);
        };
        
        _fmacro(_label, _appsec(%_TEXT, _pop(), ":\n"));
        _fmacro(_asminit, {
            _appsec(%_PRE, "global _start\n");
        });
        _fmacro(_alloc, {
            _appsec(%_DATA, _pop(), ": db ", _pop(), "\n");
        });
        _fmacro(_syscall, {
            _appsec(%_TEXT, "mov rax, ", _pop(), "\n");
            _appsec(%_TEXT, "syscall\n");
        });
        _fmacro(_sysprintn, {
            _appsec(%_TEXT, "mov rsi, ", _pop(), "\n");
            _appsec(%_TEXT, "mov rdx, ", _pop(), "\n");
            _appsec(%_TEXT, "mov rdi, 1\n");
            _syscall(1);
        });
        _fmacro(_null, {});
        _fmacro(_loop, {
            _appsec(%_TEXT, "mov rcx, ", _pop(), "\n");
            _appsec(%_TEXT, _pop(), ":\n");
            _pop();
            _appsec(%_TEXT, "dec rcx\n");
            _appsec(%_TEXT, "cmp rcx, 0\n");
            _appsec(%_TEXT, "jnz L0\n");
        });
        
        _asminit();
        _label("_start");
        _alloc(hw_str, "'Hello, World!', 10");
        _sysprintn(hw_str, 14);
        _syscall(60);
    """
    lexer = Lexer(input)
    lexer.lex()
    lexer.print()

    parser = Parser(lexer.tokens)
    tree = parser.parse(None)
    parser.print_tree(tree)

    pp = PreProcess(lexer.tokens, tree)
    pp.pp()
    
    asm_file = open("output.asm", "w")
    asm_file.write(pp.cg.output())
    asm_file.close()
main()
