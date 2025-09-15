import re

def parse_expression(expression):
    return re.findall(r'\d+|[\+\-\*\/\(\)]', expression)

def apply_operation(tokens, i):
    left = int(tokens[i-1])
    right = int(tokens[i+1])
    if tokens[i] == '*':
        result = left * right
    elif tokens[i] == '/':
        result = left / right
    elif tokens[i] == '+':
        result = left + right
    elif tokens[i] == '-':
        result = left - right
    return tokens[:i-1] + [str(int(result))] + tokens[i+2:]

def evaluate_expression(tokens):
    while '(' in tokens:
        open_index = -1
        for i in range(len(tokens)):
            if tokens[i] == '(':
                open_index = i
            elif tokens[i] == ')':
                sub_result = evaluate_expression(tokens[open_index + 1:i])
                tokens = tokens[:open_index] + sub_result + tokens[i + 1:]
                break
    steps = [tokens]
    step_count = 1
    while steps:
        next_steps = []
        results = []
        for current_tokens in steps:
            # 先处理乘除法
            found = False
            for i in range(len(current_tokens)):
                if current_tokens[i] in '*/':
                    new_tokens = apply_operation(current_tokens, i)
                    new_expression = ''.join(new_tokens)
                    next_steps.append(new_tokens)
                    results.append(f"  {new_expression}")
                    found = True

            # 如果没有乘除法，再处理加减法
            if not found:
                for i in range(len(current_tokens)):
                    if current_tokens[i] in '+-':
                        new_tokens = apply_operation(current_tokens, i)
                        new_expression = ''.join(new_tokens)
                        next_steps.append(new_tokens)
                        results.append(f"  {new_expression}")
                        # 确保每次只处理一个运算符
                        break
        if not results:
            break
        steps = next_steps
        step_count += 1
    return steps[0]

def generate_steps(expression):
    tokens = parse_expression(expression)
    steps = [tokens]
    step_count = 1

    while steps:
        next_steps = []
        results = []

        for current_tokens in steps:
            # 处理括号
            if '(' in current_tokens:
                for i in range(len(current_tokens)):
                    if current_tokens[i] == '(':
                        open_index = i
                    if current_tokens[i] == ')':
                        close_index = i
                        sub_expression = current_tokens[open_index + 1:close_index]
                        sub_result = evaluate_expression(sub_expression)
                        new_tokens = current_tokens[:open_index] + sub_result + current_tokens[close_index + 1:]
                        new_expression = ''.join(new_tokens)
                        next_steps.append(new_tokens)
                        results.append(f"  {new_expression}")
                        break
            else:
                # 先处理乘除法
                found = False
                for i in range(len(current_tokens)):
                    if current_tokens[i] in '*/':
                        new_tokens = apply_operation(current_tokens, i)
                        new_expression = ''.join(new_tokens)
                        next_steps.append(new_tokens)
                        results.append(f"  {new_expression}")
                        found = True

                # 如果没有乘除法，再处理加减法
                if not found:
                    for i in range(len(current_tokens)):
                        if current_tokens[i] in '+-':
                            new_tokens = apply_operation(current_tokens, i)
                            new_expression = ''.join(new_tokens)
                            next_steps.append(new_tokens)
                            results.append(f"  {new_expression}")
                            # 确保每次只处理一个运算符
                            break

        if not results:
            break

        print(f"\n第{step_count}步运算可得到：")
        for result in results:
            print(result)

        steps = next_steps
        step_count += 1

# 测试
expression = "(200-2)*48-4*12"
generate_steps(expression)
