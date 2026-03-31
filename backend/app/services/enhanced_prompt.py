from typing import List, Dict, Any, Optional
from datetime import datetime
import re


class EnhancedPromptBuilder:
    """增强的Prompt构建器，支持多种模板和优化策略"""

    def __init__(self, template_type: str = "default"):
        """
        初始化Prompt构建器

        Args:
            template_type: 模板类型，支持 default, strict, creative, technical
        """
        self.template_type = template_type
        self.system_templates = {
            "default": self._default_system_prompt,
            "strict": self._strict_system_prompt,
            "creative": self._creative_system_prompt,
            "technical": self._technical_system_prompt,
            "concise": self._concise_system_prompt,
            "detailed": self._detailed_system_prompt
        }

        # 加载模板
        self.system_prompt = self.system_templates.get(template_type, self._default_system_prompt)()

        # 上下文模板
        self.context_templates = {
            "default": self._default_context_template,
            "detailed": self._detailed_context_template,
            "minimal": self._minimal_context_template
        }

        # 用户提示模板
        self.user_templates = {
            "default": self._default_user_template,
            "step_by_step": self._step_by_step_user_template,
            "with_examples": self._with_examples_user_template
        }

        print(f"✅ Prompt构建器初始化成功，模板类型: {template_type}")

    def _default_system_prompt(self) -> str:
        """默认系统提示"""
        return """你是一个本地知识库问答助手。请严格遵循以下规则：

1. **基于上下文回答**：只使用提供的上下文信息，不要编造或添加外部知识
2. **明确不确定性**：如果上下文不足以回答问题，明确说"根据当前知识库内容无法确定"
3. **引用来源**：在回答中引用使用的片段编号，如[片段1][片段3]
4. **保持简洁**：回答要简洁明了，避免冗长
5. **语言一致**：使用与问题相同的语言回答
6. **格式规范**：回答使用Markdown格式，列表使用-，代码使用```"""

    def _strict_system_prompt(self) -> str:
        """严格模式系统提示"""
        return """你是一个严格的本地知识库问答助手。请严格遵循以下规则：

1. **绝对基于上下文**：只能使用提供的上下文，任何外部知识都是禁止的
2. **明确拒绝**：如果问题超出上下文范围，必须明确拒绝回答
3. **精确引用**：必须精确引用每个使用的片段
4. **避免推测**：不要进行任何推测或假设
5. **事实核查**：确保回答的每个事实都有上下文支撑
6. **格式要求**：回答必须是纯文本，不使用任何格式"""

    def _creative_system_prompt(self) -> str:
        """创意模式系统提示"""
        return """你是一个有创造力的本地知识库问答助手。请遵循以下规则：

1. **基于上下文**：主要使用提供的上下文，但可以适当推理
2. **启发式回答**：当信息不足时，基于上下文进行合理推理
3. **多角度思考**：从不同角度分析问题
4. **结构清晰**：回答要有良好的结构和逻辑
5. **举例说明**：适当使用例子来解释概念
6. **鼓励探索**：鼓励用户进一步探索相关问题"""

    def _technical_system_prompt(self) -> str:
        """技术模式系统提示"""
        return """你是一个技术文档问答助手。请严格遵循以下规则：

1. **技术准确性**：确保技术术语和概念的准确性
2. **代码处理**：如果涉及代码，确保格式正确且可执行
3. **版本敏感**：注意技术文档的版本信息
4. **依赖关系**：明确技术组件的依赖关系
5. **最佳实践**：遵循技术最佳实践
6. **错误处理**：提供完整的技术错误处理方案"""

    def _concise_system_prompt(self) -> str:
        """简洁模式系统提示"""
        return """你是一个简洁的本地知识库问答助手。请遵循以下规则：

1. **简洁回答**：用最少的文字回答问题
2. **重点突出**：突出最重要的信息
3. **避免冗余**：避免重复和冗余信息
4. **直接回答**：直接回答问题，不绕弯子
5. **关键词**：使用关键词而不是长句
6. **结构化**：使用列表和要点组织信息"""

    def _detailed_system_prompt(self) -> str:
        """详细模式系统提示"""
        return """你是一个详细的本地知识库问答助手。请遵循以下规则：

1. **全面回答**：尽可能全面地回答问题
2. **详细解释**：详细解释每个概念和步骤
3. **背景信息**：提供相关的背景信息
4. **多种角度**：从多个角度分析问题
5. **深入细节**：深入技术细节和实现原理
6. **扩展知识**：适当扩展相关知识点"""

    def _default_context_template(self, hits: List[Dict]) -> str:
        """默认上下文模板"""
        context_lines = []
        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            context_lines.append(
                f"[片段{i}] 来源={meta.get('doc_name')} | chunk={meta.get('chunk_index')}\n"
                f"{hit['content']}"
            )
        return "\n\n".join(context_lines) if context_lines else "无可用上下文"

    def _detailed_context_template(self, hits: List[Dict]) -> str:
        """详细上下文模板"""
        context_lines = []
        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            context_lines.append(
                f"--- 片段{i} 开始 ---\n"
                f"来源：{meta.get('doc_name')} | 文档ID：{meta.get('doc_id')} | 块索引：{meta.get('chunk_index')}\n"
                f"内容：{hit['content']}\n"
                f"相关度：{hit.get('score', 0):.3f}\n"
                f"--- 片段{i} 结束 ---\n"
            )
        return "\n".join(context_lines) if context_lines else "无可用上下文"

    def _minimal_context_template(self, hits: List[Dict]) -> str:
        """最小化上下文模板"""
        context_lines = []
        for i, hit in enumerate(hits, start=1):
            context_lines.append(f"[片段{i}] {hit['content']}")
        return "\n".join(context_lines) if context_lines else "无可用上下文"

    def _default_user_template(self, question: str, context_text: str, context_guide: str) -> str:
        """默认用户提示模板"""
        return f"""### 问题：
{question}

### 可用上下文：
{context_text}

{context_guide if context_text != "无可用上下文" else ""}

### 回答要求：
1. **基于上下文**：只使用提供的上下文信息
2. **引用来源**：在回答中明确引用片段编号，如[片段1]
3. **处理不确定性**：如果信息不足，明确说明
4. **语言一致性**：使用与问题相同的语言
5. **格式规范**：使用Markdown格式，列表使用-"""

    def _step_by_step_user_template(self, question: str, context_text: str, context_guide: str) -> str:
        """分步回答用户提示模板"""
        return f"""### 问题：
{question}

### 可用上下文：
{context_text}

{context_guide if context_text != "无可用上下文" else ""}

### 回答要求（分步进行）：
1. **分析问题**：先分析问题的关键点和需求
2. **查找信息**：在上下文中查找相关信息
3. **整理思路**：整理回答的思路和大纲
4. **撰写回答**：按照大纲撰写回答
5. **检查引用**：确保每个事实都有来源引用
6. **格式优化**：优化回答的格式和可读性"""

    def _with_examples_user_template(self, question: str, context_text: str, context_guide: str) -> str:
        """带示例的用户提示模板"""
        return f"""### 问题：
{question}

### 可用上下文：
{context_text}

{context_guide if context_text != "无可用上下文" else ""}

### 回答要求：
1. **概念解释**：先解释相关的概念和术语
2. **提供示例**：使用具体的例子来说明
3. **实践应用**：说明如何在实际中应用
4. **注意事项**：提供使用时的注意事项
5. **最佳实践**：给出最佳实践建议
6. **引用来源**：在回答中明确引用片段编号"""

    def build_messages(
        self,
        question: str,
        history: List[Dict],
        hits: List[Dict],
        context_template: str = "default",
        user_template: str = "default",
        max_context_length: int = 4000,
        include_context_guide: bool = True
    ) -> List[Dict]:
        """
        构建增强的prompt消息

        Args:
            question: 用户问题
            history: 对话历史
            hits: 检索到的上下文片段
            context_template: 上下文模板类型
            user_template: 用户提示模板类型
            max_context_length: 最大上下文长度
            include_context_guide: 是否包含上下文使用指南

        Returns:
            消息列表
        """
        # 智能选择上下文
        selected_hits = self._select_context_smart(hits, max_context_length)

        # 构建上下文文本
        context_template_func = self.context_templates.get(context_template, self._default_context_template)
        context_text = context_template_func(selected_hits)

        # 构建上下文指南
        context_guide = ""
        if include_context_guide and context_text != "无可用上下文":
            context_guide = self._build_context_guide(selected_hits)

        # 构建用户提示
        user_template_func = self.user_templates.get(user_template, self._default_user_template)
        user_prompt = user_template_func(question, context_text, context_guide)

        # 构建消息列表
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加对话历史（如果有）
        if history:
            limited_history = self._limit_history(history)
            for msg in limited_history:
                if msg.get("role") in {"user", "assistant"}:
                    messages.append(msg)

        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _select_context_smart(self, hits: List[Dict], max_length: int) -> List[Dict]:
        """智能选择上下文，考虑长度和相关性"""
        if not hits:
            return []

        # 按相关度排序
        sorted_hits = sorted(hits, key=lambda x: x.get('score', 0), reverse=True)

        selected = []
        current_length = 0

        for hit in sorted_hits:
            content_length = len(hit.get('content', ''))
            if current_length + content_length <= max_length:
                selected.append(hit)
                current_length += content_length
            else:
                # 如果单个片段太大，尝试截断
                if not selected:  # 还没有选择任何片段
                    truncated_hit = hit.copy()
                    truncated_hit['content'] = hit['content'][:max_length]
                    selected.append(truncated_hit)
                break

        return selected

    def _build_context_guide(self, hits: List[Dict]) -> str:
        """构建上下文使用指南"""
        return f"""
### 上下文使用指南：
1. **片段数量**：{len(hits)}个相关片段
2. **使用规则**：必须明确引用使用的片段编号
3. **优先级**：相关度高的片段优先使用
4. **组合策略**：可以组合多个片段的信息
5. **不确定性处理**：如果信息冲突，说明不确定性
"""

    def _limit_history(self, history: List[Dict], max_history: int = 5) -> List[Dict]:
        """限制历史消息数量"""
        if len(history) <= max_history:
            return history

        # 保留最近的max_history条消息
        return history[-max_history:]

    def build_few_shot_prompt(
        self,
        question: str,
        examples: List[Dict],
        hits: List[Dict],
        context_template: str = "default",
        max_context_length: int = 4000
    ) -> List[Dict]:
        """
        构建few-shot prompt，包含示例

        Args:
            question: 用户问题
            examples: 示例列表，每个示例包含question和answer
            hits: 检索到的上下文片段
            context_template: 上下文模板类型
            max_context_length: 最大上下文长度

        Returns:
            消息列表
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加示例
        for example in examples:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})

        # 添加当前问题的上下文
        selected_hits = self._select_context_smart(hits, max_context_length)
        context_template_func = self.context_templates.get(context_template, self._default_context_template)
        context_text = context_template_func(selected_hits)

        user_prompt = f"""### 问题：
{question}

### 可用上下文：
{context_text}

### 回答要求：
请参考上述示例，基于提供的上下文回答问题。确保引用来源，保持回答的准确性和一致性。"""

        messages.append({"role": "user", "content": user_prompt})
        return messages


class PromptOptimizer:
    """Prompt优化器，用于优化和改进prompt效果"""

    def __init__(self):
        self.optimization_strategies = {
            "reduce_hallucination": self._reduce_hallucination,
            "improve_clarity": self._improve_clarity,
            "enhance_accuracy": self._enhance_accuracy,
            "optimize_length": self._optimize_length
        }

    def _reduce_hallucination(self, prompt: str) -> str:
        """减少幻觉的策略"""
        anti_hallucination_rules = """
1. **严格基于上下文**：只使用提供的上下文信息，不要添加任何外部知识
2. **明确不确定性**：如果信息不足，明确说明"根据当前知识库内容无法确定"
3. **避免推测**：不要进行任何推测、假设或推断
4. **精确引用**：每个事实都必须有明确的来源引用
5. **检查一致性**：确保回答与上下文信息一致
6. **拒绝外部知识**：即使你知道更多信息，也要严格基于上下文
"""
        return prompt + anti_hallucination_rules

    def _improve_clarity(self, prompt: str) -> str:
        """提高清晰度的策略"""
        clarity_rules = """
1. **结构化回答**：使用清晰的标题和列表组织内容
2. **简洁明了**：避免冗长和复杂的句子
3. **重点突出**：突出最重要的信息
4. **逻辑连贯**：确保回答的逻辑流程清晰
5. **定义术语**：对专业术语进行简要解释
6. **示例说明**：使用具体例子帮助理解
"""
        return prompt + clarity_rules

    def _enhance_accuracy(self, prompt: str) -> str:
        """提高准确性的策略"""
        accuracy_rules = """
1. **事实核查**：确保回答的每个事实都有上下文支撑
2. **版本敏感**：注意文档的版本信息和时效性
3. **精确引用**：准确引用片段编号，不混淆
4. **避免绝对化**：使用"可能"、"通常"等适当的限定词
5. **多角度验证**：从多个角度验证信息的准确性
6. **承认局限性**：承认知识库的局限性
"""
        return prompt + accuracy_rules

    def _optimize_length(self, prompt: str, target_length: int = 1000) -> str:
        """优化长度的策略"""
        if len(prompt) <= target_length:
            return prompt

        # 简单的长度优化：移除冗余的空白和过长的描述
        optimized = re.sub(r'\n\s*\n', '\n\n', prompt)  # 移除多余的空白行
        optimized = re.sub(r' {2,}', ' ', optimized)  # 移除多余的空格

        if len(optimized) > target_length:
            # 如果还是太长，截断用户提示部分
            parts = optimized.split("### 回答要求：")
            if len(parts) > 1:
                system_part = parts[0]
                user_part = parts[1][:target_length - len(system_part)]
                optimized = system_part + "### 回答要求：" + user_part

        return optimized

    def optimize(self, prompt: str, strategies: List[str] = None, target_length: int = 1000) -> str:
        """
        优化prompt

        Args:
            prompt: 原始prompt
            strategies: 优化策略列表
            target_length: 目标长度

        Returns:
            优化后的prompt
        """
        if strategies is None:
            strategies = ["reduce_hallucination", "improve_clarity", "enhance_accuracy"]

        optimized_prompt = prompt
        for strategy in strategies:
            if strategy in self.optimization_strategies:
                optimized_prompt = self.optimization_strategies[strategy](optimized_prompt)

        # 最后优化长度
        optimized_prompt = self._optimize_length(optimized_prompt, target_length)

        return optimized_prompt


# 测试代码
if __name__ == "__main__":
    # 测试不同模板
    print("=== 测试不同Prompt模板 ===")

    # 默认模板
    builder_default = EnhancedPromptBuilder("default")
    hits = [
        {"content": "Python是一种流行的编程语言", "metadata": {"doc_name": "Python介绍", "chunk_index": 0}, "score": 0.9},
        {"content": "Java是另一种广泛使用的编程语言", "metadata": {"doc_name": "Java教程", "chunk_index": 1}, "score": 0.8}
    ]
    history = [{"role": "user", "content": "之前的问题"}]

    messages_default = builder_default.build_messages("什么是Python？", history, hits)
    print(f"\n默认模板消息数: {len(messages_default)}")
    print(f"系统提示长度: {len(messages_default[0]['content'])}")

    # 严格模板
    builder_strict = EnhancedPromptBuilder("strict")
    messages_strict = builder_strict.build_messages("什么是Python？", history, hits)
    print(f"\n严格模板消息数: {len(messages_strict)}")
    print(f"系统提示关键词: 严格、精确、拒绝")

    # 技术模板
    builder_tech = EnhancedPromptBuilder("technical")
    messages_tech = builder_tech.build_messages("如何安装Python？", history, hits)
    print(f"\n技术模板消息数: {len(messages_tech)}")
    print(f"系统提示关键词: 技术、准确、代码")

    # 测试Prompt优化
    print("\n=== 测试Prompt优化 ===")
    optimizer = PromptOptimizer()

    original_prompt = """
你是一个AI助手。请回答用户的问题。

### 问题：
什么是Python？

### 可用上下文：
Python是一种流行的编程语言。

### 回答要求：
1. 基于上下文回答
2. 保持简洁
3. 使用中文
"""

    optimized_prompt = optimizer.optimize(
        original_prompt,
        strategies=["reduce_hallucination", "improve_clarity"],
        target_length=800
    )

    print(f"原始prompt长度: {len(original_prompt)}")
    print(f"优化后prompt长度: {len(optimized_prompt)}")
    print(f"优化策略: 减少幻觉 + 提高清晰度")

    # 测试few-shot prompt
    print("\n=== 测试Few-shot Prompt ===")
    examples = [
        {"question": "什么是Java？", "answer": "Java是一种广泛使用的编程语言，特别适用于企业级应用。[片段2]"},
        {"question": "Python有什么特点？", "answer": "Python以简洁易读的语法著称，适合初学者学习。[片段1]"}
    ]

    few_shot_messages = builder_default.build_few_shot_prompt(
        "什么是编程语言？", examples, hits
    )
    print(f"Few-shot消息数: {len(few_shot_messages)}")
    print(f"示例数量: {len(examples)}")