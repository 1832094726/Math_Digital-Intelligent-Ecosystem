import re

# 解析表达式，找出所有的运算符、数字和括号（括号不算运算符）
def parse_expression(expression):
    tokens = re.findall(r'\d+|[-+*/()]', expression)
    return tokens

# 计算表达式中的运算符数量（不包括括号）
def count_operators(expression):
    # 只统计加减乘除运算符，括号不计入
    return len(re.findall(r'[-+*/]', expression))

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

    possible_steps = []

    # 处理括号中的表达式
    if '(' in tokens:
        start = tokens.index('(')
        end = len(tokens) - tokens[::-1].index(')') - 1
        inside_expr = tokens[start + 1:end]
        inside_result = evaluate_inside_parentheses(inside_expr)
        new_tokens = tokens[:start] + [inside_result] + tokens[end + 1:]
        new_expression = ''.join(new_tokens)
        generate_steps(new_expression, steps.copy(), all_paths)
        return all_paths

    # 处理乘除运算
    for i in range(1, len(tokens) - 1, 2):
        if tokens[i] in '*/':
            new_tokens = calculate_step(tokens, i)
            new_expression = ''.join(new_tokens)
            possible_steps.append(new_expression)

    # 如果没有乘除运算，处理加减法运算
    if not possible_steps:
        for i in range(1, len(tokens) - 1, 2):
            if tokens[i] in '+-':
                new_tokens = calculate_step(tokens, i)
                new_expression = ''.join(new_tokens)
                possible_steps.append(new_expression)
                break

    # 递归处理每个可能的步骤
    for step in possible_steps:
        generate_steps(step, steps.copy(), all_paths)

    return all_paths

# 计算一步操作后的新表达式
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

# 计算括号内部表达式的值
def evaluate_inside_parentheses(tokens):
    expression = ''.join(tokens)
    return str(eval(expression))  # 使用 eval 计算括号内部的表达式

# 递归获取标准路径中执行n次操作后的表达式
def get_expression_after_n_steps(std_path, n):
    if n < len(std_path):
        return std_path[n]
    return None

def analyze_student_solution(student_solution, standard_solutions):
    student_steps = student_solution.split('=')

    print("\n学生解答步骤：")
    prev_step = student_steps[0]
    prev_operators = count_operators(prev_step)

    all_correct = True  # 用于标记是否所有步骤都正确

    for idx in range(1, len(student_steps)):
        curr_step = student_steps[idx]
        curr_operators = count_operators(curr_step)
        operators_reduced = prev_operators - curr_operators

        print(f"第{idx}步：学生解答 '{curr_step}'，减少了 {operators_reduced} 个运算符")

        # 寻找标准答案中完全匹配的步骤，而不仅仅是减少运算符的数量
        found_match = False
        for std_path in standard_solutions:
            # 寻找标准路径中与当前学生解答完全相同的表达式
            if curr_step in std_path:
                found_match = True
                print(f"匹配的标准步骤：'{curr_step}'")
                break

        if not found_match:
            all_correct = False
            print(f"未找到匹配的标准步骤。学生解答在第{idx}步出错。")
            # 不退出，继续检查后续步骤

        prev_step = curr_step
        prev_operators = curr_operators

    if all_correct:
        print("学生解答正确！")
    else:
        print("学生解答中有错误。")

    return all_correct


# 测试代码
expression = "200-2*(48-4)+1*12"
all_paths = generate_steps(expression)

print("\n生成的所有标准解答路径：")
for path in all_paths:
    print(path)

# 学生解答过程
# student_solution = "200-2*(48-4)+1*12=200-2*44+12=200-88+12=124"
# student_solution = "200-2*(48-4)+1*12=200-88+1*12=200-88+12=124"
student_solution = "200-2*(48-4)+1*12=200-88+1*12=200-88+12=10+12=124"

# 分析学生解答与标准答案的对比
analyze_student_solution(student_solution, all_paths)
