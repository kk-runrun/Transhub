import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from openai import OpenAI
# 引入之前分离的逻辑模块
from backend import FileParser, SimpleRAG, AIAgent

# ==========================================
# 0. 全局配置 (白名单)
# ==========================================
# 在这里设置允许注册的人员名单 (只有这些用户名能注册成功)
ALLOWED_USERS_WHITELIST = ["admin", "manager_li", "translator_01", "dev_test"]

# ==========================================
# 1. 核心应用逻辑 (保持原样不动)
# ==========================================
st.set_page_config(layout="wide", page_title="AI 批量翻译工作台")

def main_app():
    """
    这里是原来的核心功能代码，只有登录成功才会执行到这里
    """
    # --- 原有代码开始 ---
    if "tasks" not in st.session_state:
        st.session_state.tasks = {} 
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False

    with st.sidebar:
        st.title("⚙️ 设置与输入")
        # 显示当前登录用户
        st.info(f"当前用户: {st.session_state.get('name', 'Unknown')}")
        
        st.session_state.api_key = st.text_input("API Key", type="password")
        
        if st.session_state.api_key:
            try:
                client = OpenAI(api_key=st.session_state.api_key)
                agent = AIAgent(client)
            except Exception as e:
                st.error(f"API 连接失败: {e}")
                st.stop()
        else:
            st.warning("请输入 API Key")
            st.stop()

        kb_files = st.file_uploader("上传知识库 (RAG)", accept_multiple_files=True)
        if kb_files and st.button("建立索引"):
            rag = SimpleRAG(client)
            rag.ingest(kb_files)
            st.session_state.rag_system = rag
            st.success("知识库建立完成")

        target_files = st.file_uploader("待翻译文件", accept_multiple_files=True)
        if target_files and st.button("🚀 开始"):
            for f in target_files:
                if f.name not in st.session_state.tasks:
                    content = FileParser.extract_text(f) 
                    content = FileParser.optimize_text(content)
                    st.session_state.tasks[f.name] = {
                        "status": "翻译中", 
                        "raw": content, 
                        "chunks": [content],
                        "translated_chunks": [],
                        "result": None,
                        "api_log": []
                    }
            st.session_state.processing = True
            st.rerun()

    st.title("🏭 智能翻译工作台")
    
    if st.session_state.get("processing"):
        for name, task in st.session_state.tasks.items():
            if task['status'] == "翻译中":
                # 简单调用演示
                if 'agent' in locals():
                    res = agent.run_translation(task['chunks'][0], "", "翻译它")
                    task['result'] = res
                    task['status'] = "已完成"
                else:
                    st.error("Agent 未初始化")
        st.session_state.processing = False
        st.rerun()

    if st.session_state.tasks:
        selected = st.selectbox("选择文件", list(st.session_state.tasks.keys()))
        task = st.session_state.tasks[selected]
        c1, c2 = st.columns(2)
        with c1: st.text_area("原文", task['raw'])
        with c2: 
            if task['result']:
                st.write(task['result'])
    # --- 原有代码结束 ---

# ==========================================
# 2. 门卫逻辑 (Gatekeeper)
# ==========================================
if __name__ == "__main__":
    # A. 加载配置文件
# A. 加载配置文件
    # 强制使用 utf-8 读取
    with open('config.yaml', encoding='utf-8') as file:
        config = yaml.load(file, Loader=SafeLoader)

    # B. 初始化认证器
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

# C. 显示登录窗口
    # 1. 只负责渲染界面，不接收返回值
    authenticator.login(location='main')
    
    # 2. 手动从缓存中提取状态
    authentication_status = st.session_state.get('authentication_status')
    name = st.session_state.get('name')
    username = st.session_state.get('username')

    # D. 判断登录状态
    if authentication_status:
        # --- 登录成功 ---
        # 在 Sidebar 显示退出按钮
        authenticator.logout('退出登录', 'sidebar')
        st.session_state['name'] = name # 存入 session 方便后续使用
        st.session_state['username'] = username
        
        # 打开大门，运行主程序
        main_app()

    elif authentication_status is False:
        # --- 密码错误 ---
        st.error('用户名或密码错误')

    elif authentication_status is None:
        # --- 未登录 / 注册入口 ---
        st.warning('请登录以继续使用')
        
        st.markdown("---")
        # 自定义注册逻辑
        with st.expander("📝 没有账号？点此注册"):
            new_user = st.text_input("设置用户名 (ID)")
            new_name = st.text_input("设置昵称 (Display Name)")
            new_pass = st.text_input("设置密码", type="password")
            new_pass2 = st.text_input("确认密码", type="password")
            
            if st.button("提交注册"):
                # 1. 校验白名单
                if new_user not in ALLOWED_USERS_WHITELIST:
                    st.error("🚫 用户无权限，不可使用 (用户名不在白名单内)")
                # 2. 校验是否已存在
                elif new_user in config['credentials']['usernames']:
                    st.warning("⚠️ 该用户已存在，请直接登录")
                # 3. 校验密码一致性
                elif new_pass != new_pass2:
                    st.error("❌ 两次输入的密码不一致")
                elif not new_pass:
                    st.error("❌ 密码不能为空")
                else:
                    # 4. 注册成功：哈希加密并写入文件
                    try:
                        # 生成哈希密码
                        hashed_pass = stauth.Hasher([new_pass]).generate()[0]
                        
                        # 更新 config 对象
                        config['credentials']['usernames'][new_user] = {
                            "name": new_name,
                            "password": hashed_pass
                        }
                        
                        # 写入 yaml 文件
                        with open('config.yaml', 'w', encoding='utf-8') as file:
                            yaml.dump(config, file, default_flow_style=False)
                            
                        st.success(f"✅ 注册成功！用户 [{new_user}] 已添加，请在上方登录。")
                    except Exception as e:
                        st.error(f"注册写入失败: {e}")