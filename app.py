import streamlit as st
import pandas as pd
from docx import Document
from openai import OpenAI
from openai import APIError
import json
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# 1. 核心工具类 (Backend Logic)
# ==========================================

class FileParser:
    """处理文件读取与导出"""
    @staticmethod
    def extract_text(uploaded_file):
        """从 Excel/Word/Txt 提取纯文本"""
        filename = uploaded_file.name.lower()
        content = ""
        try:
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
                # 将 DataFrame 转为文本，按行拼接
                content = df.to_string(index=False)
            elif filename.endswith(('.docx', '.doc')):
                doc = Document(uploaded_file)
                content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            else: # txt
                content = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            return f"Error reading file: {str(e)}"
        return content

    @staticmethod
    def optimize_text(text):
        """优化文本：去除 NaN 和英文"""
        # The user provided a very long specific string.
        nan_string_to_remove = "NaN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             NaN                                                                                                                                                                                                                                                                                                                                                                                                               NaN           NaN                    NaN                                            NaN        NaN                                                                                                                                                                     NaN                                                                                                                                                    NaN"
        text = text.replace(nan_string_to_remove, "")
        # Also use a more general regex to clean up other similar patterns
        text = re.sub(r'(NaN\s*)+', '', text)
        # Remove all English characters (a-z, A-Z)
        text = re.sub(r'[a-zA-Z]', '', text)
        return text

    @staticmethod
    def generate_word(text):
        """将文本写入内存中的 Word 文件"""
        doc = Document()
        for line in text.split('\n'):
            if line.strip():
                doc.add_paragraph(line)
        bio = io.BytesIO()
        doc.save(bio)
        bio.seek(0)
        return bio

class SimpleRAG:
    """简易 RAG 引擎：处理知识库嵌入与检索"""
    def __init__(self, client, embedding_model="text-embedding-3-small"):
        self.client = client
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = []
        
    def ingest(self, files):
        """处理上传的知识库文件"""
        self.chunks = []
        raw_text = ""
        for f in files:
            raw_text += FileParser.extract_text(f) + "\n"
        
        # 简单切片：按 500 字符切分 (实际项目可使用更高级的 TextSplitter)
        step = 2000
        self.chunks = [raw_text[i:i+step] for i in range(0, len(raw_text), step) if raw_text[i:i+step].strip()]
        
        if self.chunks:
            # 批量生成 Embedding
            response = self.client.embeddings.create(
                input=self.chunks,
                model=self.embedding_model
            )
            self.embeddings = [item.embedding for item in response.data]

    def retrieve(self, query, top_k=3):
        """检索最相关的知识"""
        if not self.embeddings:
            return "（未上传知识库，无背景信息）"
            
        # 生成查询向量
        q_resp = self.client.embeddings.create(input=[query], model=self.embedding_model)
        q_vec = q_resp.data[0].embedding
        
        # 计算余弦相似度
        sim_matrix = cosine_similarity([q_vec], self.embeddings)
        top_indices = np.argsort(sim_matrix[0])[::-1][:top_k]
        
        results = [self.chunks[i] for i in top_indices]
        return "\n---\n".join(results)

class AIAgent:
    """处理所有 LLM 交互"""
    def __init__(self, client, model_name="gpt-4o"):
        self.client = client
        self.model = model_name

    def run_translation(self, text, context, prompt, api_log=None):
        """仅执行双路翻译"""
        sys_prompt = f"""
        你是一个翻译引擎。
        【背景知识】: {context}
        【用户指令】: {prompt}
        请生成两个版本的翻译：
        1. version_precise: 忠实原文，术语精准，直译为主。
        2. version_fluent: 本土化表达，营销口吻，流畅自然。
        必须返回严格的 JSON 格式: {{"v1": "...", "v2": "..."}}
        """
        if api_log is not None:
            api_log.append(f"--- Translation Request ---\n[Prompt]\n{sys_prompt}\n\n[Text Chunk]\n{text[:500]}...")
        try:
            # 尝试调用 API
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"待翻译文本:\n{text}\n\n{sys_prompt}"}],
                response_format={"type": "json_object"}
            )
            # 正常处理响应（res 此时应为 ChatCompletion 对象）
            response_content = res.choices[0].message.content
            if api_log is not None:
                api_log.append(f"--- Translation Response ---\n{response_content}")
            return json.loads(response_content)
        
        # 【重要修改点 2】: 捕获特定的 OpenAI 异常
        except APIError as e:
            error_message = f"OpenAI API Error (Code {e.status_code}): {e.message}"
            if api_log is not None:
                api_log.append(f"--- Translation ERROR ---\n{error_message}")
            return {"error": error_message}
            
        # 【重要修改点 3】: 捕获其他通用错误，包括 json.JSONDecodeError 和前面提到的 AttributeError
        except Exception as e:
            error_message = f"General Error: {str(e)}. Response type: {type(res) if 'res' in locals() else 'N/A'}"
            if api_log is not None:
                api_log.append(f"--- Translation ERROR ---\n{error_message}")
            return {"error": error_message}

    def run_review(self, text, context, trans_data, api_log=None):
        # ... (此方法保持不变，但为了健壮性，建议也修改其异常处理块)
        judge_prompt = f"""
        【背景】: {context}
        【原文】: {text}
        【版本1 (精准)】: {trans_data.get('v1')}
        【版本2 (流畅)】: {trans_data.get('v2')}
        请对比两者，选出更好的版本，并给出理由。
        必须返回 JSON: {{"best_version": "v1" 或 "v2", "reason": "...", "suggestion": "改进建议..."}}
        """
        if api_log is not None:
            api_log.append(f"--- Review Request ---\n[Prompt]\n{judge_prompt}")
        try:
            judge_res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"}
            )
            response_content = judge_res.choices[0].message.content
            if api_log is not None:
                api_log.append(f"--- Review Response ---\n{response_content}")
            return json.loads(response_content)
        
        except APIError as e:
            error_message = f"OpenAI API Error (Code {e.status_code}): {e.message}"
            if api_log is not None:
                api_log.append(f"--- Review ERROR ---\n{error_message}")
            return {"best_version": "v1", "reason": f"API Error: {error_message}", "suggestion": ""}

        except Exception as e:
            error_message = f"General Error: {str(e)}"
            if api_log is not None:
                api_log.append(f"--- Review ERROR ---\n{error_message}")
            return {"best_version": "v1", "reason": f"API Error: {error_message}", "suggestion": ""}

    def run_qa_check(self, text, rules):
        # ... (此方法也建议修改异常处理)
        sys_prompt = f"""
        你是一个严格的 QA 质检员。
        【质检标准】: {rules}
        
        请检查用户提交的文本。
        如果通过，status 为 "PASS"。
        如果不通过，status 为 "FAIL"，并提供 reason（原因）和 fix_suggestion（修改建议）。
        
        返回 JSON: {{"status": "PASS" / "FAIL", "reason": "...", "fix_suggestion": "..."}}
        """
        
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"待检文本: {text}\n\n{sys_prompt}"}],
                response_format={"type": "json_object"}
            )
            return json.loads(res.choices[0].message.content)
        
        except APIError as e:
            error_message = f"OpenAI API Error (Code {e.status_code}): {e.message}"
            return {"status": "FAIL", "reason": f"QA Check API Error: {error_message}", "fix_suggestion": "无法运行 QA 检查。"}

        except Exception as e:
            error_message = f"General Error: {str(e)}"
            return {"status": "FAIL", "reason": f"QA Check General Error: {error_message}", "fix_suggestion": "无法运行 QA 检查。"}

# ==========================================
# 2. 前端界面 (Streamlit UI)
# ==========================================

st.set_page_config(layout="wide", page_title="AI 批量翻译 + QA 工作流")

# Session State 初始化
if "tasks" not in st.session_state:
    st.session_state.tasks = {} # {filename: {status, raw, result, final, qa_log}}
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_base" not in st.session_state:
    st.session_state.api_base = "https://api.openai.com/v1"
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-4o"
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = "text-embedding-3-small"
if "api_timeout" not in st.session_state:
    st.session_state.api_timeout = 300
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- 侧边栏：配置与上传 ---
with st.sidebar:
    st.title("⚙️ 设置与输入")
    
    # 1. API 配置
    st.session_state.api_key = st.text_input(
        "API Key",
        value=st.session_state.get("api_key", ""),
        type="password"
    )
    st.session_state.api_base = st.text_input(
        "API Base URL",
        value=st.session_state.get("api_base", "https://api.openai.com/v1")
    )
    st.session_state.model_name = st.text_input(
        "Chat Model Name",
        value=st.session_state.get("model_name", "gpt-4o")
    )
    st.session_state.embedding_model = st.text_input(
        "Embedding Model Name",
        value=st.session_state.get("embedding_model", "text-embedding-3-small")
    )
    st.session_state.api_timeout = st.number_input(
        "API Timeout (seconds)",
        min_value=30,
        max_value=600,
        value=st.session_state.get("api_timeout", 300),
        step=10,
        help="设置调用AI API的超时时间（秒）。如果处理大文件时遇到“Request timed out”错误，请尝试增加此值。"
    )
        
    if not st.session_state.api_key:
        st.warning("请输入 API Key 以继续")
        st.stop()
        
    try:
        client = OpenAI(api_key=st.session_state.api_key, base_url=st.session_state.api_base, timeout=st.session_state.api_timeout)
        agent = AIAgent(client, model_name=st.session_state.model_name)
    except Exception as e:
        st.error(f"API 连接失败: {e}")
        st.stop()

    # 2. 知识库 (RAG)
    st.markdown("### 📚 1. 知识库 (可选)")
    kb_files = st.file_uploader("上传产品手册/术语表", accept_multiple_files=True, key="kb")
    if kb_files and st.button("建立知识库索引"):
        with st.spinner("正在向量化知识库..."):
            rag = SimpleRAG(client, embedding_model=st.session_state.embedding_model)
            rag.ingest(kb_files)
            st.session_state.rag_system = rag
        st.success(f"知识库已就绪 ({len(rag.chunks)} chunks)")

    # 3. 待翻译文件
    st.markdown("### 📄 2. 待翻译文件")
    target_files = st.file_uploader("上传 Excel/Word/Txt", accept_multiple_files=True, key="target")
    optimize_file = st.checkbox("优化文件 (去除 NaN 和英文内容)")
    
    # 4. 提示词配置
    st.markdown("### 💬 3. 提示词配置")
    trans_prompt = st.text_area("翻译提示词", "翻译为简体中文，保持专业性。")
    qa_prompt = st.text_area("QA 质检标准 (可编辑)", 
                             "检查是否有错别字、关键术语错误或格式错乱。如果一切正常，请通过。")

    # 5. 批量开始按钮
    if target_files and st.button("🚀 开始批量翻译"):
        # 初始化任务
        for f in target_files:
            if f.name not in st.session_state.tasks:
                content = FileParser.extract_text(f)
                if optimize_file:
                    content = FileParser.optimize_text(content)
                # NEW: Chunking
                chunk_size = 8000 # characters
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size) if content[i:i+chunk_size].strip()]

                st.session_state.tasks[f.name] = {
                    "status": "等待处理", # 状态机: 等待处理 -> 翻译中 -> 评审中 -> 待人工审核 -> QA检查中 -> 需返工 -> 已完成
                    "raw": content,
                    "chunks": chunks,
                    "translated_chunks": [],
                    "result": None,
                    "final": "",
                    "qa_feedback": None,
                    "api_log": []
                }
        st.session_state.processing = True
        st.rerun()

# --- 主界面：工作台 ---

st.title("🏭 智能翻译工作台")

# --- NEW: Chunk-based, multi-stage, asynchronous-style processing loop ---
if st.session_state.get("processing", False):
    # Find the next task to process
    active_task_name = None
    active_task = None
    for name, task_data in st.session_state.tasks.items():
        if task_data['status'] in ["等待处理", "翻译中", "评审中"]:
            active_task_name = name
            active_task = task_data
            break # Process one file at a time

    if not active_task:
        st.session_state.processing = False
        st.success("所有文件处理完成！")
        st.rerun()
    else:
        current_status = active_task['status']
        
        # --- Progress Bar Logic ---
        total_chunks_all_files = sum(len(t.get('chunks', [])) for t in st.session_state.tasks.values())
        completed_chunks_all_files = sum(len(t.get('translated_chunks', [])) for t in st.session_state.tasks.values())
        if total_chunks_all_files > 0:
            progress = completed_chunks_all_files / total_chunks_all_files
            progress_text = f"文件: {active_task_name} | 状态: {current_status} | 总进度: {completed_chunks_all_files}/{total_chunks_all_files} 块"
            st.progress(progress, text=progress_text)

        # --- RAG Context ---
        context = ""
        if st.session_state.rag_system:
            context = st.session_state.rag_system.retrieve(active_task['raw'][:500])

        # --- State Machine ---
        if current_status == "等待处理":
            st.session_state.tasks[active_task_name]['status'] = "翻译中"
            st.rerun()

        elif current_status == "翻译中":
            chunk_index = len(active_task['translated_chunks'])
            if chunk_index < len(active_task['chunks']):
                chunk_to_translate = active_task['chunks'][chunk_index]
                trans_data = agent.run_translation(chunk_to_translate, context, trans_prompt, api_log=active_task['api_log'])
                
                if trans_data.get("error"):
                    st.error(f"文件 {active_task_name} 块 {chunk_index+1} 翻译失败: {trans_data['error']}")
                    st.session_state.tasks[active_task_name]['status'] = "需返工"
                    st.session_state.processing = False # Stop processing on error
                    st.rerun()
                else:
                    st.session_state.tasks[active_task_name]['translated_chunks'].append(trans_data)
                    st.rerun()
            else: # All chunks translated
                v1_full = "".join([c.get('v1', '') for c in active_task['translated_chunks']])
                v2_full = "".join([c.get('v2', '') for c in active_task['translated_chunks']])
                st.session_state.tasks[active_task_name]['result'] = {
                    "v1": v1_full, "v2": v2_full, "context_used": context
                }
                st.session_state.tasks[active_task_name]['status'] = "评审中"
                st.rerun()

        elif current_status == "评审中":
            full_trans_data = active_task.get('result', {})
            eval_data = agent.run_review(active_task['raw'], context, full_trans_data, api_log=active_task['api_log'])
            
            st.session_state.tasks[active_task_name]['result']['eval'] = eval_data
            st.session_state.tasks[active_task_name]['status'] = "待人工审核"
            best_ver_key = eval_data.get('best_version', 'v1')
            st.session_state.tasks[active_task_name]['final'] = full_trans_data.get(best_ver_key, "")
            st.rerun()

if not st.session_state.tasks:
    st.info("👈 请在左侧上传文件并点击“开始批量翻译”")
else:
    # 任务列表导航
    task_names = list(st.session_state.tasks.keys())
    
    # 自定义格式显示状态
    def format_func(name):
        status = st.session_state.tasks[name]['status']
        icons = {
            "等待处理": "⚪", "翻译中": "-", "评审中": "--",
            "待人工审核": "🟠", "QA检查中": "🔵",
            "需返工": "🔴", "已完成": "✅"
        }
        return f"{icons.get(status, '❓')} {name} - [{status}]"

    selected_file = st.selectbox("选择文件进行处理:", task_names, format_func=format_func)
    current_task = st.session_state.tasks[selected_file]

    if current_task['status'] == "等待处理":
        st.warning("该文件尚未翻译，请点击左侧“开始批量翻译”。")
    else:
        # ================= 布局：三栏设计 =================
        col_left, col_mid, col_right = st.columns([1, 1.2, 1.5])

        # --- 左栏：参考信息 ---
        with col_left:
            st.subheader("1. 原文与背景")
            with st.expander("查看原文", expanded=True):
                st.text_area("Raw Text", current_task['raw'], height=300, disabled=True)
            with st.expander("📚 知识库背景 (RAG)", expanded=False):
                if current_task['result']:
                    st.info(current_task['result'].get('context_used', '无'))

    # --- 中栏：AI 对比 ---
            with col_mid:
                st.subheader("2. AI 双路翻译")
                res = current_task['result']
                
                # 【修复点】先检查 res 是否有效，以及是否有 error 字段
                if res and "error" in res and res["error"]:
                    st.error(f"❌ 处理失败: {res['error']}")
                    st.info("请检查 API Key 或网络连接，然后点击左侧重新开始。")
                
                # 【修复点】使用 .get() 安全读取 eval，防止 KeyError
                elif res and res.get('eval'):
                    # 显示评审意见
                    eval_data = res['eval']
                    best_ver = eval_data.get('best_version', '未知')
                    
                    st.success(f"🏆 AI 推荐: **{best_ver}**")
                    st.caption(f"理由: {eval_data.get('reason', '无')}")
                    if eval_data.get('suggestion'):
                        st.warning(f"建议: {eval_data.get('suggestion')}")
                    
                    # Tab 切换查看版本
                    tab1, tab2 = st.tabs(["版本 1 (精准)", "版本 2 (流畅)"])
                    with tab1: st.write(res.get('v1', '无内容'))
                    with tab2: st.write(res.get('v2', '无内容'))
                
                else:
                    st.info("等待生成结果...")

        # --- 右栏：人工编辑 + QA ---
        with col_right:
            st.subheader("3. 最终确认与 QA")
            
            # 状态提示条
            status = current_task['status']
            if status == "需返工":
                st.error(f"❌ QA 未通过: {current_task['qa_feedback']}")
            elif status == "已完成":
                st.success("✅ QA 通过，准备交付！")

            # 编辑区域
            is_locked = (status == "已完成")
            edited_text = st.text_area(
                "在此编辑最终译文", 
                value=current_task['final'], 
                height=400,
                disabled=is_locked
            )

            # 操作按钮区
            c1, c2 = st.columns(2)
            
            # 按钮逻辑：提交 QA
            with c1:
                if not is_locked:
                    if st.button("🕵️ 提交 AI 质检 (QA)", key="btn_qa"):
                        with st.spinner("AI 质检员正在审核..."):
                            # 调用 QA Agent
                            qa_res = agent.run_qa_check(edited_text, qa_prompt)
                            
                            if qa_res['status'] == "PASS":
                                st.session_state.tasks[selected_file]['status'] = "已完成"
                                st.session_state.tasks[selected_file]['final'] = edited_text
                                st.session_state.tasks[selected_file]['qa_feedback'] = "Passed"
                                st.balloons()
                                st.rerun()
                            else:
                                st.session_state.tasks[selected_file]['status'] = "需返工"
                                st.session_state.tasks[selected_file]['qa_feedback'] = f"{qa_res.get('reason')} \n\n建议: {qa_res.get('fix_suggestion')}"
                                st.session_state.tasks[selected_file]['final'] = edited_text # 保存当前的修改
                                st.rerun()

            # 按钮逻辑：强制通过
            with c2:
                if not is_locked and status == "需返工":
                    if st.button("⚠️ 忽略报错，强制通过"):
                        st.session_state.tasks[selected_file]['status'] = "已完成"
                        st.session_state.tasks[selected_file]['final'] = edited_text
                        st.rerun()
            
            # 按钮逻辑：重置
            if is_locked:
                if st.button("🔄 撤销完成状态 (重新编辑)"):
                    st.session_state.tasks[selected_file]['status'] = "待人工审核"
                    st.rerun()

            # API 日志
            with st.expander("🔍 查看 API 日志"):
                if current_task.get('api_log'):
                    log_content = "\n\n".join(current_task['api_log'])
                    st.code(log_content, language='text')
                else:
                    st.info("暂无日志。")

            # 导出下载
            if status == "已完成":
                st.markdown("---")
                st.write("📥 **下载文件:**")
                # Word 下载
                docx = FileParser.generate_word(edited_text)
                st.download_button(
                    label="Word (.docx)",
                    data=docx,
                    file_name=f"trans_{selected_file}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                # Txt 下载
                st.download_button(
                    label="Text (.txt)",
                    data=edited_text,
                    file_name=f"trans_{selected_file}.txt",
                    mime="text/plain"
                )