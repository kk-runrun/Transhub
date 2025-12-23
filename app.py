import streamlit as st
from openai import OpenAI
# 【重要】引入分离出去的逻辑模块
from backend import FileParser, SimpleRAG, AIAgent

# 设置页面必须在最开头，不能放在函数里
st.set_page_config(layout="wide", page_title="AI 批量翻译工作台")

def main_app():
    """
    这里包裹原来的所有前端代码
    只有登录成功后，才会运行这个函数
    """
    # -----------------------------------------------------
    # 原来的 Session State 初始化
    # -----------------------------------------------------
    if "tasks" not in st.session_state:
        st.session_state.tasks = {} 
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    # ... (其他 state 初始化建议保留，为了精简这里省略，实际运行需加上) ...
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # -----------------------------------------------------
    # 原来的 Sidebar 代码
    # -----------------------------------------------------
    with st.sidebar:
        st.title("⚙️ 设置与输入")
        st.session_state.api_key = st.text_input("API Key", type="password")
        
        # 初始化 Client 和 Agent
        if st.session_state.api_key:
            try:
                client = OpenAI(api_key=st.session_state.api_key)
                agent = AIAgent(client) # 使用 backend 中的类
            except Exception as e:
                st.error(f"API 连接失败: {e}")
                st.stop()
        else:
            st.warning("请输入 API Key")
            st.stop()

        # 知识库部分
        kb_files = st.file_uploader("上传知识库 (RAG)", accept_multiple_files=True)
        if kb_files and st.button("建立索引"):
            rag = SimpleRAG(client) # 使用 backend 中的类
            rag.ingest(kb_files)
            st.session_state.rag_system = rag
            st.success("知识库建立完成")

        # 翻译文件部分
        target_files = st.file_uploader("待翻译文件", accept_multiple_files=True)
        if target_files and st.button("🚀 开始"):
            for f in target_files:
                if f.name not in st.session_state.tasks:
                    # 使用 backend 中的类
                    content = FileParser.extract_text(f) 
                    content = FileParser.optimize_text(content)
                    # 简化逻辑演示
                    st.session_state.tasks[f.name] = {
                        "status": "翻译中", 
                        "raw": content, 
                        "chunks": [content], # 简单处理
                        "translated_chunks": [],
                        "result": None,
                        "api_log": []
                    }
            st.session_state.processing = True
            st.rerun()

    # -----------------------------------------------------
    # 原来的主界面代码
    # -----------------------------------------------------
    st.title("🏭 智能翻译工作台")
    
    # 简单的处理循环演示 (逻辑与之前一致，调用 agent)
    if st.session_state.get("processing"):
        for name, task in st.session_state.tasks.items():
            if task['status'] == "翻译中":
                # 调用 agent
                res = agent.run_translation(task['chunks'][0], "", "翻译它")
                task['result'] = res
                task['status'] = "已完成"
        st.session_state.processing = False
        st.rerun()

    # 结果展示
    if st.session_state.tasks:
        selected = st.selectbox("选择文件", list(st.session_state.tasks.keys()))
        task = st.session_state.tasks[selected]
        c1, c2 = st.columns(2)
        with c1: st.text_area("原文", task['raw'])
        with c2: 
            if task['result']:
                st.write(task['result'])

# ==========================================
# 程序入口
# ==========================================
if __name__ == "__main__":
    # 暂时直接运行主程序，下一步我们会在这里加 "门卫"
    main_app()