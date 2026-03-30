import streamlit as st
import requests

st.set_page_config(page_title="本地知识库问答", page_icon="📚", layout="wide")
st.title("📚 本地知识库智能问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
    if st.button("上传并入库", use_container_width=True):
        if not files:
            st.warning("请先选择文件")
        else:
            multipart = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
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
            r = requests.post(f"{api_base}/documents/web", json={"url": url.strip()}, timeout=120)
            if r.ok:
                st.success("网页入库成功")
            else:
                st.error(r.text)

    st.divider()
    st.subheader("文档管理")
    if st.button("刷新文档列表", use_container_width=True):
        st.rerun()

    doc_resp = requests.get(f"{api_base}/documents", timeout=30)
    if doc_resp.ok:
        docs = doc_resp.json().get("documents", [])
        if not docs:
            st.caption("暂无文档")
        else:
            for d in docs:
                with st.expander(f"{d['doc_name']} ({d['chunk_count']} chunks)"):
                    st.caption(d["source"])
                    if st.button(f"删除 {d['doc_id'][:8]}", key=f"del_{d['doc_id']}"):
                        dr = requests.delete(f"{api_base}/documents/{d['doc_id']}", timeout=30)
                        if dr.ok:
                            st.success("删除成功")
                            st.rerun()
                        else:
                            st.error(dr.text)
    else:
        st.error("无法获取文档列表，请检查后端服务。")

st.subheader("对话问答")
streaming = st.checkbox("启用流式输出（仅回答，不含来源展示）", value=False)

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
    st.session_state.messages.append({"role": "user", "content": question})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in {"user", "assistant"}
    ]

    with st.chat_message("assistant"):
        if streaming:
            placeholder = st.empty()
            full = ""
            with requests.post(
                f"{api_base}/chat/stream",
                json={"question": question, "history": history},
                stream=True,
                timeout=300
            ) as r:
                if not r.ok:
                    st.error(r.text)
                else:
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full += chunk
                            placeholder.markdown(full)
            st.session_state.messages.append({"role": "assistant", "content": full})
        else:
            r = requests.post(
                f"{api_base}/chat",
                json={"question": question, "history": history},
                timeout=300
            )
            if not r.ok:
                st.error(r.text)
            else:
                data = r.json()
                st.markdown(data["answer"])
                with st.expander("查看召回来源"):
                    for i, s in enumerate(data["sources"], start=1):
                        st.markdown(f"**{i}. {s['doc_name']}** | chunk={s['chunk_index']} | score={s['score']:.4f}")
                        st.caption(s["content"][:400] + ("..." if len(s["content"]) > 400 else ""))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"]
                })