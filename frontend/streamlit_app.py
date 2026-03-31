import streamlit as st
import requests
import json

st.set_page_config(page_title="本地知识库问答", page_icon="📚", layout="wide")
st.title("📚 本地知识库智能问答系统")

# 会话状态初始化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

if "document_filters" not in st.session_state:
    st.session_state.document_filters = {}

with st.sidebar:
    st.header("配置")
    api_base = st.text_input("后端地址", "http://127.0.0.1:8000")
    st.caption("建议先启动 FastAPI 再使用本页面。")

    st.divider()
    st.subheader("上传文档")
    files = st.file_uploader(
        "支持 PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    # 文档上传配置
    with st.expander("上传配置"):
        tags = st.text_input("标签（用逗号分隔）", "")
        category = st.text_input("分类", "")

    if st.button("上传并入库", use_container_width=True):
        if not files:
            st.warning("请先选择文件")
        else:
            # 准备标签数据
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            multipart = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
            if tag_list:
                multipart.append(("tags", ",".join(tag_list)))
            if category:
                multipart.append(("category", category))

            r = requests.post(f"{api_base}/documents/upload", files=multipart, timeout=120)
            if r.ok:
                st.success("上传成功")
            else:
                st.error(r.text)

    st.divider()
    st.subheader("网页入库")
    url = st.text_input("网页 URL")
    if st.button("抓取网页并入库", use_container_width=True):
        if not url.strip():
            st.warning("请输入URL")
        else:
            # 准备网页抓取配置
            web_data = {"url": url.strip()}
            if tags:
                web_data["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if category:
                web_data["category"] = category

            r = requests.post(f"{api_base}/documents/web", json=web_data, timeout=120)
            if r.ok:
                st.success("网页入库成功")
            else:
                st.error(r.text)

    st.divider()
    st.subheader("文档管理")

    # 文档搜索和过滤
    with st.expander("文档搜索"):
        search_query = st.text_input("搜索关键词")
        filter_category = st.text_input("按分类过滤")
        filter_tags = st.text_input("按标签过滤（用逗号分隔）")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("搜索文档", use_container_width=True):
                st.session_state.document_filters = {
                    "q": search_query,
                    "category": filter_category,
                    "tags": filter_tags
                }
                st.rerun()

        with col2:
            if st.button("显示全部", use_container_width=True):
                st.session_state.document_filters = {}
                st.rerun()

    if st.button("刷新文档列表", use_container_width=True):
        st.rerun()

    # 获取文档列表（支持过滤）
    params = {}
    if st.session_state.document_filters:
        if st.session_state.document_filters.get("q"):
            params["q"] = st.session_state.document_filters["q"]
        if st.session_state.document_filters.get("category"):
            params["category"] = st.session_state.document_filters["category"]
        if st.session_state.document_filters.get("tags"):
            params["tags"] = [tag.strip() for tag in st.session_state.document_filters["tags"].split(",") if tag.strip()]

    doc_resp = requests.get(f"{api_base}/documents", params=params, timeout=30)
    if doc_resp.ok:
        docs = doc_resp.json().get("documents", [])
        if not docs:
            st.caption("暂无文档")
        else:
            st.caption(f"找到 {len(docs)} 个文档")
            for d in docs:
                # 显示文档信息
                tags_str = " ".join([f"🏷️{tag}" for tag in d.get('tags', [])])
                with st.expander(f"{d['doc_name']} ({d['chunk_count']} chunks)"):
                    st.caption(f"来源: {d['source']} | 分类: {d.get('category', '无')}")
                    if tags_str:
                        st.caption(f"标签: {tags_str}")

                    # 文档操作按钮
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"查看详情 {d['doc_id'][:8]}", key=f"view_{d['doc_id']}", use_container_width=True):
                            st.session_state.current_doc_id = d['doc_id']
                            st.rerun()

                    with col2:
                        if st.button(f"更新文档 {d['doc_id'][:8]}", key=f"update_{d['doc_id']}", use_container_width=True):
                            st.session_state.editing_doc = d['doc_id']
                            st.rerun()

                    with col3:
                        if st.button(f"删除 {d['doc_id'][:8]}", key=f"del_{d['doc_id']}", use_container_width=True):
                            dr = requests.delete(f"{api_base}/documents/{d['doc_id']}", timeout=30)
                            if dr.ok:
                                st.success("删除成功")
                                st.rerun()
                            else:
                                st.error(dr.text)
    else:
        st.error("无法获取文档列表，请检查后端服务。")

    # 显示文档详情
    if "current_doc_id" in st.session_state:
        with st.expander("文档详情", expanded=True):
            doc_id = st.session_state.current_doc_id
            doc_resp = requests.get(f"{api_base}/documents/{doc_id}", timeout=30)
            if doc_resp.ok:
                doc = doc_resp.json()
                st.json(doc)
            else:
                st.error("无法获取文档详情")

    # 文档更新表单
    if "editing_doc" in st.session_state:
        with st.expander("更新文档", expanded=True):
            doc_id = st.session_state.editing_doc
            doc_resp = requests.get(f"{api_base}/documents/{doc_id}", timeout=30)
            if doc_resp.ok:
                doc = doc_resp.json()

                with st.form("update_document"):
                    new_name = st.text_input("文档名称", value=doc['doc_name'])
                    new_category = st.text_input("分类", value=doc.get('category', ''))
                    new_tags = st.text_input("标签（用逗号分隔）", value=",".join(doc.get('tags', [])))

                    if st.form_submit_button("更新文档"):
                        update_data = {}
                        if new_name != doc['doc_name']:
                            update_data['doc_name'] = new_name
                        if new_category != doc.get('category', ''):
                            update_data['category'] = new_category
                        if new_tags != ",".join(doc.get('tags', [])):
                            update_data['tags'] = [tag.strip() for tag in new_tags.split(",") if tag.strip()]

                        if update_data:
                            r = requests.put(f"{api_base}/documents/{doc_id}", json=update_data, timeout=30)
                            if r.ok:
                                st.success("文档更新成功")
                                del st.session_state.editing_doc
                                st.rerun()
                            else:
                                st.error(r.text)
                        else:
                            st.info("没有检测到更改")

            else:
                st.error("无法获取文档信息")

st.subheader("对话问答")

# 对话管理
with st.expander("对话管理"):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("新建对话", use_container_width=True):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.rerun()

    with col2:
        # 加载对话列表
        conv_resp = requests.get(f"{api_base}/conversations", timeout=30)
        if conv_resp.ok:
            conversations = conv_resp.json()
            if conversations:
                conv_options = {conv['title']: conv['conv_id'] for conv in conversations}
                selected_conv = st.selectbox("选择对话历史", options=list(conv_options.keys()))
                if selected_conv:
                    conv_id = conv_options[selected_conv]
                    if st.button("加载对话", use_container_width=True):
                        st.session_state.current_conversation = conv_id

                        # 加载对话消息
                        msg_resp = requests.get(f"{api_base}/conversations/{conv_id}/messages", timeout=30)
                        if msg_resp.ok:
                            messages = msg_resp.json().get('messages', [])
                            st.session_state.messages = []
                            for msg in messages:
                                if msg['role'] in ['user', 'assistant']:
                                    st.session_state.messages.append({
                                        'role': msg['role'],
                                        'content': msg['content'],
                                        'sources': msg.get('sources', [])
                                    })
                        st.rerun()

    with col3:
        if st.session_state.current_conversation:
            conv_id = st.session_state.current_conversation
            if st.button("删除当前对话", use_container_width=True):
                dr = requests.delete(f"{api_base}/conversations/{conv_id}", timeout=30)
                if dr.ok:
                    st.success("对话删除成功")
                    st.session_state.current_conversation = None
                    st.session_state.messages = []
                    st.rerun()
                else:
                    st.error(dr.text)

# 流式输出配置
st.subheader("输出配置")
streaming_format = st.selectbox(
    "流式输出格式",
    options=["标准", "增强", "JSON", "SSE"],
    index=0,
    help="标准：纯文本；增强：包含来源信息；JSON：结构化数据；SSE：服务器推送事件"
)

streaming = streaming_format != "标准"

# 当前对话信息
if st.session_state.current_conversation:
    st.info(f"当前对话：{st.session_state.current_conversation}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("查看召回来源"):
                for i, s in enumerate(msg["sources"], start=1):
                    st.markdown(f"**{i}. {s['doc_name']}** | chunk={s['chunk_index']} | score={s['score']:.4f}")
                    st.caption(s["content"][:400] + ("..." if len(s["content"]) > 400 else ""))

question = st.chat_input("输入你的问题...")
if question:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})

    # 准备历史消息
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in {"user", "assistant"}
    ]

    # 确定使用的端点
    if st.session_state.current_conversation:
        # 使用对话端点
        conv_id = st.session_state.current_conversation
        if streaming:
            if streaming_format == "增强":
                endpoint = f"{api_base}/conversations/{conv_id}/chat/stream"
            elif streaming_format == "JSON":
                endpoint = f"{api_base}/chat/stream/json"
            elif streaming_format == "SSE":
                endpoint = f"{api_base}/chat/stream/sse"
            else:
                endpoint = f"{api_base}/conversations/{conv_id}/chat/stream"
        else:
            endpoint = f"{api_base}/conversations/{conv_id}/chat"
    else:
        # 使用普通端点
        if streaming:
            if streaming_format == "增强":
                endpoint = f"{api_base}/chat/stream/enhanced"
            elif streaming_format == "JSON":
                endpoint = f"{api_base}/chat/stream/json"
            elif streaming_format == "SSE":
                endpoint = f"{api_base}/chat/stream/sse"
            else:
                endpoint = f"{api_base}/chat/stream"
        else:
            endpoint = f"{api_base}/chat"

    with st.chat_message("assistant"):
        if streaming:
            placeholder = st.empty()
            full = ""
            sources = []

            try:
                with requests.post(
                    endpoint,
                    json={"question": question, "history": history},
                    stream=True,
                    timeout=300
                ) as r:
                    if not r.ok:
                        st.error(f"请求失败: {r.text}")
                    else:
                        # 处理不同类型的流式输出
                        if streaming_format == "JSON":
                            for line in r.iter_lines():
                                if line:
                                    line_str = line.decode('utf-8')
                                    if line_str.startswith("data: "):
                                        json_str = line_str[6:]
                                        if json_str:
                                            data = json.loads(json_str)
                                            if "choices" in data and data["choices"]:
                                                content = data["choices"][0].get("delta", {}).get("content", "")
                                                if content:
                                                    full += content
                                                    placeholder.markdown(full)
                        elif streaming_format == "SSE":
                            for line in r.iter_lines():
                                if line:
                                    line_str = line.decode('utf-8')
                                    if line_str.startswith("event: "):
                                        event_type = line_str[7:]
                                        if event_type == "content":
                                            # 处理内容事件
                                            pass
                                    elif line_str.startswith("data: "):
                                        json_str = line_str[6:]
                                        if json_str:
                                            data = json.loads(json_str)
                                            content = data.get("data", {}).get("content", "")
                                            if content:
                                                full += content
                                                placeholder.markdown(full)
                        else:
                            # 标准流式输出
                            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                                if chunk:
                                    full += chunk
                                    placeholder.markdown(full)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"流式输出错误: {e}")

        else:
            # 非流式输出
            try:
                r = requests.post(
                    endpoint,
                    json={"question": question, "history": history},
                    timeout=300
                )
                if not r.ok:
                    st.error(f"请求失败: {r.text}")
                else:
                    data = r.json()
                    st.markdown(data["answer"])

                    # 显示来源信息
                    with st.expander("查看召回来源"):
                        for i, s in enumerate(data["sources"], start=1):
                            st.markdown(f"**{i}. {s['doc_name']}** | chunk={s['chunk_index']} | score={s['score']:.4f}")
                            st.caption(s["content"][:400] + ("..." if len(s["content"]) > 400 else ""))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"]
                    })

                    # 如果启用了对话，保存消息到对话
                    if st.session_state.current_conversation:
                        conv_id = st.session_state.current_conversation
                        # 添加助手消息到对话
                        msg_data = {
                            "role": "assistant",
                            "content": data["answer"]
                        }
                        requests.post(
                            f"{api_base}/conversations/{conv_id}/messages",
                            json=msg_data,
                            timeout=30
                        )

            except Exception as e:
                st.error(f"请求错误: {e}")

# 保存消息到当前对话（如果是用户消息）
if question and st.session_state.current_conversation:
    conv_id = st.session_state.current_conversation
    msg_data = {
        "role": "user",
        "content": question
    }
    requests.post(
        f"{api_base}/conversations/{conv_id}/messages",
        json=msg_data,
        timeout=30
    )