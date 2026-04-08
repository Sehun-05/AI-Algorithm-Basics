# 构建产生式规则库
rules = [
    # 规则 1: 如果动物有毛发，那么它是哺乳动物
    (["毛发"], "哺乳动物"),
    # 规则 2: 如果动物能产奶，那么它是哺乳动物
    (["产奶"], "哺乳动物"),
    # 规则 3: 如果动物有羽毛，那么它是鸟类
    (["羽毛"], "鸟类"),
    # 规则 4: 如果动物会飞且会下蛋，那么它是鸟类
    (["会飞", "下蛋"], "鸟类"),
    # 规则 5: 如果动物是哺乳动物且吃肉，那么它是食肉动物
    (["哺乳动物", "吃肉"], "食肉动物"),
    # 规则 6: 如果动物是哺乳动物且有蹄，那么它是有蹄类动物
    (["哺乳动物", "有蹄"], "有蹄类动物"),
    # 规则 7: 如果动物是哺乳动物且是反刍动物，那么它是有蹄类动物
    (["哺乳动物", "反刍动物"], "有蹄类动物"),
    # 规则 8: 如果动物是食肉动物且是黄褐色且有黑色条纹，那么它是老虎
    (["食肉动物", "黄褐色", "黑色条纹"], "老虎"),
    # 规则 9: 如果动物是食肉动物且是黄褐色且有黑色斑点，那么它是金钱豹
    (["食肉动物", "黄褐色", "黑色斑点"], "金钱豹"),
    # 规则 10: 如果动物是有蹄类动物且有长脖子且有长腿且有暗斑点，那么它是长颈鹿
    (["有蹄类动物", "长脖子", "长腿", "暗斑点"], "长颈鹿"),
    # 规则 11: 如果动物是有蹄类动物且有黑色条纹，那么它是斑马
    (["有蹄类动物", "黑色条纹"], "斑马"),
    # 规则 12: 如果动物是鸟类且不会飞且有长脖子且有长腿，那么它是鸵鸟
    (["鸟类", "不会飞", "长脖子", "长腿"], "鸵鸟"),
    # 规则 13: 如果动物是鸟类且会游泳且不会飞且是黑白两色，那么它是企鹅
    (["鸟类", "会游泳", "不会飞", "黑白两色"], "企鹅"),
    # 规则 14: 如果动物是鸟类且会飞且是食肉动物，那么它是鹰
    (["鸟类", "善飞"], "信天翁")
]

# 正向推理机制
def forward_inference(facts):
    new_facts = set(facts)
    while True:
        added = False
        for conditions, conclusion in rules:
            if all(condition in new_facts for condition in conditions) and conclusion not in new_facts:
                new_facts.add(conclusion)
                added = True
        if not added:
            break
    return new_facts

# 逆向推理机制
def backward_inference(animal):
    required_conditions = []
    stack = [animal]
    while stack:
        current = stack.pop()
        for conditions, conclusion in rules:
            if conclusion == current:
                for condition in conditions:
                    if condition not in required_conditions:
                        required_conditions.append(condition)
                        stack.append(condition)
    return required_conditions

# 用户交互界面
def user_interface():
    print("请输入动物的特征，用逗号分隔（例如：毛发,吃肉），输入 '结束' 退出。")
    while True:
        input_str = input("输入特征：")
        if input_str == "结束":
            break
        facts = [fact.strip() for fact in input_str.split(',')]
        result = forward_inference(facts)
        animals = [animal for animal in result if animal not in facts and any(animal == rule[1] for rule in rules)]
        if animals:
            print("识别结果：", ', '.join(animals))
        else:
            print("未识别出动物。")

        animal_type = input("请输入要查询的动物类型（输入 '跳过' 跳过）：")
        if animal_type != "跳过":
            conditions = backward_inference(animal_type)
            print(f"{animal_type} 需要满足的条件：", ', '.join(conditions))

if __name__ == "__main__":
    user_interface()