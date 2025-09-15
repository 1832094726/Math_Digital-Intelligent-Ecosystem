import re

# 定义运算符的优先级
precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

# 解析表达式，找出所有的运算符、数字和括号
def parse_expression(expression):
    tokens = re.findall(r'\d+|[-+*/()]', expression)
    return tokens

# 计算表达式中的一步
def calculate_step(tokens, index):
    left_op = int(tokens[index - 1])
    operator = tokens[index]
    right_op = int(tokens[index + 1])

    if operator == '+':
        result = left_op + right_op
    elif operator == '-':
        result = left_op - right_op
    elif operator == '*':
        result = left_op * right_op
    elif operator == '/':
        result = left_op / right_op

    new_tokens = tokens[:index - 1] + [str(result)] + tokens[index + 2:]
    return new_tokens

# 计算括号内的表达式
def evaluate_parentheses(tokens):
    stack = []
    for token in tokens:
        if token == ')':
            sub_expr = []
            while stack and stack[-1] != '(':
                sub_expr.insert(0, stack.pop())
            stack.pop()  # 移除左括号 '('
            result = generate_steps(''.join(sub_expr))  # 递归计算括号内的表达式
            stack.append(str(list(result)[0][-1]))  # 拿到括号内最终的计算结果
        else:
            stack.append(token)
    return stack

# 递归生成所有步骤
def generate_steps(expression, steps=None, all_paths=None):
    if steps is None:
        steps = []
    if all_paths is None:
        all_paths = set()

    tokens = parse_expression(expression)
    steps.append(expression)

    # 如果表达式长度只有1（即计算结束），存储当前路径
    if len(tokens) == 1:
        all_paths.add(tuple(steps))
        return all_paths

    # 首先处理括号
    if '(' in tokens:
        new_tokens = evaluate_parentheses(tokens)
        new_expression = ''.join(new_tokens)
        return generate_steps(new_expression, steps.copy(), all_paths)

    possible_steps = []

    # 首先处理乘除运算，因为它们优先级更高
    for i in range(1, len(tokens) - 1, 2):
        if tokens[i] in '*/':
            new_tokens = calculate_step(tokens, i)
            new_expression = ''.join(new_tokens)
            possible_steps.append(new_expression)

    # 如果没有乘除运算，处理加减法运算
    if not possible_steps:
        for i in range(1, len(tokens) - 1, 2):
            if tokens[i] in '+-':
                # 遇到同级运算符时，优先处理左边的运算符
                if i == 1 or precedence[tokens[i - 2]] == precedence[tokens[i]]:
                    new_tokens = calculate_step(tokens, i)
                    new_expression = ''.join(new_tokens)
                    possible_steps.append(new_expression)
                    break  # 只处理左侧的运算符，避免同级计算跳过左侧

    # 递归处理每个可能的步骤
    for step in possible_steps:
        generate_steps(step, steps.copy(), all_paths)

    return all_paths
# 比较学生的步骤和标准步骤
def compare_student_solution(student_solution, standard_solutions):
    # 分析学生的解答过程
    student_steps = student_solution.split('=')

    # 检查每一步是否在标准解答的路径中，允许简化步骤
    for standard_steps in standard_solutions:
        std_idx = 0
        for stu_step in student_steps:
            while std_idx < len(standard_steps) and standard_steps[std_idx] != stu_step:
                std_idx += 1
            if std_idx == len(standard_steps):
                break  # 学生步骤不符合标准路径
        else:
            return True  # 找到匹配路径
    return False

# 测试代码
expression = "200-2*(48-4)+1*12"
all_paths = generate_steps(expression)

# 学生解答过程
student_solution = "200-2*(48-4)+1*12=200-2*44+12=200-88+12=112+12=124"
# student_solution = "200-2*(48-4)+1*12=200-2*44+12=200-86+12=122"

# 比较学生解答与标准答案
if compare_student_solution(student_solution, all_paths):
    print("学生解答正确！")
else:
    print("学生解答错误。")
