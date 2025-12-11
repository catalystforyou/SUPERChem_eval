from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd
import streamlit as st
import yaml
from openai import OpenAI
from streamlit.components.v1 import html

MAX_MODELS = 4
BASE_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_JSONL_1 = str(BASE_DATA_DIR / "20251014164938_questions_release_en_true__gemini-3-pro-preview-thinking_high__1_0_1.jsonl")
DEFAULT_JSONL_2 = str(BASE_DATA_DIR / "20251014164938_questions_release_en_true__gpt-5_high__1_0_1.jsonl")
DEFAULT_PARQUET = str(BASE_DATA_DIR / "20251014164938_questions.parquet")
DEFAULT_EVAL_PROMPT = """你是科学考试的评估员。请基于题目、官方解析（ground_truth_explanation）以及某模型的思考链（reasoning）与最终回答（output），完成以下判定：
1) 模型 reasoning+output 是否与官方解析一致，是否存在错误或遗漏。
2) 模型是否声称回忆/引用了文献、书籍、专利或编号。

请用简短中文 JSON 给出结论，字段：
- consistent: true/false
- consistency_reason: 简述一致/不一致的依据
- recalled_reference: true/false
- reference_detail: 指出模型声称引用的文献/来源片段；若无，则写 none

上下文：
Question: {question}
Reasoning: {reasoning}
Output: {output}
Ground Truth Explanation: {explanation}
"""

SUBFIELD_OPTIONS = [
    "organic chemistry",
    "inorganic chemistry",
    "physical chemistry",
    "structural chemistry",
    "analytical chemistry",
    "chemical biology",
    "polymer chemistry",
    "theoretical chemistry",
]


def extract_model_name_from_path(path: str) -> str:
    """从文件路径中提取模型名称（用__分割，取第二个元素）。"""
    filename = Path(path).stem  # 移除扩展名
    parts = filename.split("__")
    if len(parts) >= 2:
        if 'true' in filename:
            return parts[1] + ' (Multimodal)'
        else:
            return parts[1] + ' (Text-Only)'
    return filename  # 如果无法分割，返回完整文件名


def list_data_jsonls() -> List[Path]:
    """列出 data 目录下的 jsonl 文件。"""
    if not BASE_DATA_DIR.exists():
        return []
    return sorted([p for p in BASE_DATA_DIR.glob("*.jsonl") if p.is_file()])


@st.cache_resource(show_spinner=False)
def load_eval_client(target_model: str = "deepseek-chat") -> tuple[OpenAI, str]:
    """读取 view/config.yaml 并返回 OpenAI 客户端与模型名。"""
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_cfg = None
    for item in cfg.get("model_list", []):
        if item.get("model") == target_model or model_cfg is None:
            model_cfg = item
            if item.get("model") == target_model:
                break
    if not model_cfg:
        raise ValueError("配置文件中未找到可用的模型配置。")
    client = OpenAI(base_url=model_cfg["base_url"], api_key=model_cfg["api_key"])
    return client, model_cfg["model"]


def build_eval_prompt(template: str, question_row: pd.Series, model_row: pd.Series, lang: str) -> str:
    """填充评估 prompt 所需字段。"""
    question_text = question_row.get(f"question_{lang}") or question_row.get("question") or ""
    explanation_text = question_row.get(f"explanation_{lang}") or question_row.get("explanation") or ""
    reasoning_text = model_row.get("llm_reasoning") or model_row.get("llm_output") or ""
    output_text = model_row.get("llm_output") or model_row.get("llm_answer") or ""
    return template.format(
        question=str(question_text),
        reasoning=str(reasoning_text),
        output=str(output_text),
        explanation=str(explanation_text),
    )


def stream_eval(
    prompt_template: str,
    question_row: pd.Series,
    model_row: pd.Series,
    lang: str,
    placeholder,
) -> str:
    """调用 LLM 评估，流式写入 placeholder。"""
    client, model_name = load_eval_client()
    prompt = build_eval_prompt(prompt_template, question_row, model_row, lang)
    resp_text = ""
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            resp_text += delta
            placeholder.markdown(resp_text)
    return resp_text


def _read_by_ext(src: Union[str, Path, io.BytesIO], ext: str) -> pd.DataFrame:
    """按扩展名读取 jsonl/json/parquet。"""
    if ext in {".jsonl", ".json"}:
        return pd.read_json(src, lines=True)
    if ext == ".parquet":
        return pd.read_parquet(src)
    raise ValueError(f"Unsupported file type: {ext}")


@st.cache_data(show_spinner=False)
def load_from_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return _read_by_ext(path, path.suffix.lower())


@st.cache_data(show_spinner=False, hash_funcs={io.BytesIO: lambda f: f.getbuffer().tobytes()})
def load_from_upload(uploaded_file) -> pd.DataFrame:
    ext = Path(uploaded_file.name).suffix.lower()
    buffer = io.BytesIO(uploaded_file.getbuffer())
    return _read_by_ext(buffer, ext)


def pick_columns(df: pd.DataFrame, candidates: Iterable[str], fallback: int = 6) -> List[str]:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        cols = list(df.columns[:fallback])
    return cols


def image_to_data_uri(path: str, data: bytes) -> str:
    suffix = Path(path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def inject_images(md: str, img_dict: Union[dict, None]) -> str:
    """将 markdown 中的 /media/... 链接替换为 data URI。"""
    if not md or not isinstance(md, str) or not img_dict:
        return md

    def repl(match: re.Match[str]) -> str:
        url = match.group(1)
        data = img_dict.get(url)
        if data is None:
            return match.group(0)
        return f"({image_to_data_uri(url, data)})"

    return re.sub(r"\((/media/[^\s)]+)\)", repl, md)


def render_markdown_with_images(label: str, text: Union[str, None], images: Union[dict, None]) -> None:
    """展示 markdown 文本与关联图片，支持 Latex 与 data URI。"""
    if text:
        st.markdown(f"**{label}**")
        st.markdown(inject_images(str(text), images))
    extra_imgs = []
    if images:
        for url, data in images.items():
            if data is not None:
                extra_imgs.append((url, data))
    if extra_imgs:
        st.markdown(f"**{label} 图片**")
        st.image(
            [d for _, d in extra_imgs],
            caption=[u for u, _ in extra_imgs],
            width=320,
        )


def ensure_uuid(df: pd.DataFrame, desc: str) -> None:
    if "uuid" not in df.columns:
        raise KeyError(f"{desc} 缺少 uuid 列，无法对齐题目与模型输出。")


def render_options_block(label: str, value: Union[List, dict, str, None], images: Union[dict, None]) -> None:
    if value is None:
        return
    st.markdown(f"**{label}**")
    if isinstance(value, dict):
        for k, v in value.items():
            if v is None:
                continue
            text = str(v).strip()
            if not text or text.lower() == "none":
                continue
            st.markdown(f"- **{k}**. {inject_images(text, images)}")
    elif isinstance(value, list):
        for v in value:
            if v is None:
                continue
            text = str(v).strip()
            if not text or text.lower() == "none":
                continue
            st.markdown(f"- {inject_images(text, images)}")
    else:
        text = str(value).strip()
        if text and text.lower() != "none":
            st.markdown(inject_images(text, images))


def render_question_block(question_row: pd.Series, lang: str) -> None:
    question_imgs = question_row.get("question_images")
    render_markdown_with_images(
        "题干" if lang == "zh" else "Question",
        question_row.get(f"question_{lang}") or question_row.get("question"),
        question_imgs,
    )
    st.divider()

    options_imgs = question_row.get("options_images")
    render_options_block(
        "选项" if lang == "zh" else "Options",
        question_row.get(f"options_{lang}"),
        options_imgs,
    )
    st.divider()

    answer_value = question_row.get(f"answer_{lang}") or question_row.get("answer")
    if answer_value is not None:
        st.markdown("**答案**" if lang == "zh" else "**Answer**")
        st.markdown(str(answer_value))
        st.divider()

    explanation_imgs = question_row.get("explanation_images")
    render_markdown_with_images(
        "解析" if lang == "zh" else "Explanation",
        question_row.get(f"explanation_{lang}") or question_row.get("explanation"),
        explanation_imgs,
    )
    st.divider()


def main() -> None:
    st.set_page_config(page_title="SUPERChem LLM 对比检查", layout="wide")
    st.title("SUPERChem LLM 对比检查")
    st.caption("人工核查 SUPERChem 题目与多模型回答，支持最多四个模型对比。")

    with st.sidebar:
        st.header("题库 (Parquet)")
        path_parquet = st.text_input("题库路径", value=str(DEFAULT_PARQUET))
        uploaded_parquet = st.file_uploader("或上传 Parquet", type=["parquet"], key="upload_parquet")

        st.divider()
        st.subheader("展示语言")
        lang = st.radio(
            "Language",
            options=["zh", "en"],
            index=0,
            format_func=lambda x: "中文" if x == "zh" else "English",
        )

        st.divider()
        st.subheader("模型 (最多 4 个)")
        available_data_files = list_data_jsonls()
        data_options = ["(自定义路径)"] + [p.name for p in available_data_files]
        model_count = st.slider("需要对比的模型数量", min_value=1, max_value=MAX_MODELS, value=2)
        model_inputs = []
        for i in range(model_count):
            with st.expander(f"模型 {i + 1}", expanded=i < 2):
                # 根据路径或上传文件自动生成默认名称
                if i == 0:
                    default_path = str(DEFAULT_JSONL_1)
                elif i == 1:
                    default_path = str(DEFAULT_JSONL_2)
                else:
                    default_path = ""

                default_choice_index = 0
                if default_path:
                    default_name = Path(default_path).name
                    if default_name in data_options:
                        default_choice_index = data_options.index(default_name)

                file_choice = st.selectbox(
                    "选择 data 目录模型文件",
                    options=data_options,
                    index=default_choice_index,
                    key=f"data_choice_{i}",
                )

                if file_choice == "(自定义路径)":
                    model_path = st.text_input("JSONL 路径", value=default_path, key=f"path_jsonl_{i}")
                else:
                    model_path = str(BASE_DATA_DIR / file_choice)
                    st.caption(f"使用 data/{file_choice}")

                uploaded_jsonl = st.file_uploader(
                    "或上传 JSONL/JSON",
                    type=["jsonl", "json"],
                    key=f"upload_jsonl_{i}",
                )

                # 自动生成默认显示名称
                if uploaded_jsonl is not None:
                    extracted_label = extract_model_name_from_path(uploaded_jsonl.name)
                elif model_path.strip():
                    extracted_label = extract_model_name_from_path(model_path.strip())
                else:
                    extracted_label = None

                default_label = extracted_label or f"Model {i + 1}"
                model_label = st.text_input("展示名称", value=default_label, key=f"model_label_{i}")

                # 如果用户未自定义且已有提取名，优先用提取名
                final_label = model_label.strip()
                if not final_label:
                    final_label = default_label
                if extracted_label and final_label == f"Model {i + 1}":
                    final_label = extracted_label

                model_inputs.append(
                    {
                        "label": final_label,
                        "path": model_path.strip(),
                        "upload": uploaded_jsonl,
                    }
                )

        st.divider()
        st.subheader("LLM 评估 Prompt")
        eval_prompt = st.text_area(
            "用于评估 reasoning+output 与解析的一致性，可实时修改",
            value=DEFAULT_EVAL_PROMPT,
            height=220,
            key="eval_prompt",
        )

        st.divider()
        st.subheader("检索 / 过滤")
        keyword_question = st.text_input("题目全文关键词（支持uuid）", key="kw_question")
        keyword_answer = st.text_input("回答全文关键词", key="kw_answer")
        subfield_filter = st.multiselect(
            "学科领域 (subfield)",
            options=SUBFIELD_OPTIONS,
            default=SUBFIELD_OPTIONS,
            key="subfield_filter",
        )
        score_filter_box = st.container()

    # 读取题库
    try:
        if uploaded_parquet is not None:
            df_q = load_from_upload(uploaded_parquet)
            q_desc = uploaded_parquet.name
        else:
            df_q = load_from_path(path_parquet)
            q_desc = Path(path_parquet).name
        ensure_uuid(df_q, "题库文件")
    except Exception as exc:  # pragma: no cover - UI error显示
        st.error(f"读取题库失败: {exc}")
        return

    # 读取模型
    model_frames = []
    for idx, cfg in enumerate(model_inputs, start=1):
        if not cfg["path"] and cfg["upload"] is None:
            continue
        try:
            if cfg["upload"] is not None:
                df_model = load_from_upload(cfg["upload"])
                desc = cfg["upload"].name
            else:
                df_model = load_from_path(cfg["path"])
                desc = Path(cfg["path"]).name
            ensure_uuid(df_model, f"模型 {idx}")
        except Exception as exc:
            st.error(f"读取模型 {idx} 失败: {exc}")
            continue
        df_model = df_model.copy()
        df_model["__model_name__"] = cfg["label"]
        model_frames.append((cfg["label"], df_model, desc))

    if not model_frames:
        st.warning("至少需要加载一个模型的 JSONL 文件。")
        return

    uuid_sets = [set(df_q["uuid"])] + [set(df["uuid"]) for _, df, _ in model_frames]
    common_uuids = set.intersection(*uuid_sets)
    if not common_uuids:
        st.warning("题库与模型的 uuid 无交集，请确认文件是否对应同一批题目。")
        return

    question_base = df_q[df_q["uuid"].isin(common_uuids)].copy()

    long_rows: List[pd.DataFrame] = []
    score_options: dict[str, List] = {}
    for label, df_model, _desc in model_frames:
        df_model = df_model[df_model["uuid"].isin(common_uuids)].copy()
        llm_cols = ["llm_answer", "llm_reasoning", "llm_output", "score", "finish_reason", "model", "language"]
        keep_cols = ["uuid"] + [c for c in llm_cols if c in df_model.columns]
        merged = question_base.merge(df_model[keep_cols], on="uuid", how="inner")
        merged["model_name"] = label
        long_rows.append(merged)
        if "score" in df_model.columns:
            score_options[label] = sorted(df_model["score"].dropna().unique().tolist())
        else:
            score_options[label] = []

    if not long_rows:
        st.warning("没有可用的模型数据。")
        return

    combined = pd.concat(long_rows, ignore_index=True)

    with score_filter_box:
        st.subheader("按模型 score 过滤")
        intersection_only = st.checkbox("仅展示交集", value=False, key="score_intersection_only")
        model_score_filters: dict[str, List] = {}
        for label, scores in score_options.items():
            if scores:
                model_score_filters[label] = st.multiselect(
                    f"{label} score",
                    options=scores,
                    default=scores,
                    key=f"score_filter_{label}",
                )
            else:
                st.caption(f"{label}: 无 score 列")

    # 过滤逻辑
    mask = pd.Series(True, index=combined.index)
    question_cols = [
        c
        for c in [
            f"question_{lang}",
            f"options_{lang}",
            f"explanation_{lang}",
            f"answer_{lang}",
            "question",
            "prompt",
        ]
        if c in combined.columns
    ]
    if keyword_question:
        search_fields = {
            "uuid": combined["uuid"].astype(str).str.contains(keyword_question, case=False, na=False)
        }
        if question_cols:
            search_fields.update(
                {c: combined[c].astype(str).str.contains(keyword_question, case=False, na=False) for c in question_cols}
            )
        contains = pd.DataFrame(search_fields)
        mask &= contains.any(axis=1)

    answer_cols = [c for c in ["llm_answer", "llm_reasoning", "llm_output"] if c in combined.columns]
    if keyword_answer and answer_cols:
        contains = pd.DataFrame(
            {c: combined[c].astype(str).str.contains(keyword_answer, case=False, na=False) for c in answer_cols}
        )
        mask &= contains.any(axis=1)

    # subfield 过滤
    if subfield_filter and "subfield" in combined.columns:
        mask &= combined["subfield"].isin(subfield_filter)

    for label, scores in model_score_filters.items():
        if not scores:
            continue
        has_score_col = "score" in combined.columns
        if not has_score_col:
            continue
        current_mask = (combined["model_name"] == label) & (~combined["score"].isin(scores))
        mask &= ~current_mask

    filtered = combined[mask].reset_index(drop=True)
    if intersection_only:
        required_count = len(model_frames)
        uuid_counts = filtered.groupby("uuid")["model_name"].nunique()
        keep_uuids = uuid_counts[uuid_counts == required_count].index
        filtered = filtered[filtered["uuid"].isin(keep_uuids)].reset_index(drop=True)
    st.success(
        f"题库 {len(df_q)} 行，模型共 {sum(len(df) for _, df, _ in model_frames)} 行，"
        f"uuid 交集 {len(common_uuids)}，过滤后 {len(filtered)} 行"
    )

    if filtered.empty:
        st.warning("过滤后无数据，请调整筛选条件。")
        return

    if "eval_results" not in st.session_state:
        st.session_state.eval_results = {}

    # 摘要表：每个 uuid 的各模型 score
    meta_cols = [
        c
        for c in ["difficulty", "knowledge_type", "subfield", "tags", "source", f"answer_{lang}"]
        if c in question_base.columns
    ]
    summary_base = question_base.set_index("uuid")[meta_cols]
    score_pivot = filtered.pivot_table(index="uuid", columns="model_name", values="score", aggfunc="first")
    summary_df = summary_base.join(score_pivot)
    st.markdown("### 概览")
    st.dataframe(summary_df.reset_index(), use_container_width=True, height=360)

    # 统计信息
    st.markdown("### 统计信息")
    unique_uuids = filtered["uuid"].nunique()
    st.write(f"**当前筛选下题目数量**: {unique_uuids}")

    if "score" in filtered.columns:
        stats_data = []
        for label, _, _ in model_frames:
            model_data = filtered[filtered["model_name"] == label]
            if not model_data.empty:
                total = len(model_data)
                correct = (model_data["score"] == 1).sum()
                wrong = (model_data["score"] == 0).sum()
                accuracy = (correct / total * 100) if total > 0 else 0
                stats_data.append({
                    "模型": label,
                    "正确数": correct,
                    "错误数": wrong,
                    "正确率": f"{accuracy:.2f}%",
                })
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # 详情
    st.markdown("### 记录详情")
    uuid_options = filtered["uuid"].unique().tolist()
    if not uuid_options:
        st.warning("无可展示的 uuid。")
        return

    if "uuid_idx" not in st.session_state:
        st.session_state.uuid_idx = 0
    # 当前索引兜底
    st.session_state.uuid_idx = min(st.session_state.uuid_idx, len(uuid_options) - 1)

    def sync_select_from_idx() -> None:
        st.session_state.uuid_select_value = uuid_options[st.session_state.uuid_idx]

    # 初始化 select value
    if "uuid_select_value" not in st.session_state:
        st.session_state.uuid_select_value = uuid_options[st.session_state.uuid_idx]

    def on_select_change() -> None:
        val = st.session_state.uuid_select_value
        if val in uuid_options:
            st.session_state.uuid_idx = uuid_options.index(val)

    nav_cols = st.columns([1, 2, 1])
    with nav_cols[0]:
        prev_clicked = st.button("⬅️ 上一题", key="btn_prev")
    with nav_cols[2]:
        next_clicked = st.button("下一题 ➡️", key="btn_next")

    if prev_clicked:
        st.session_state.uuid_idx = (st.session_state.uuid_idx - 1) % len(uuid_options)
        sync_select_from_idx()
    if next_clicked:
        st.session_state.uuid_idx = (st.session_state.uuid_idx + 1) % len(uuid_options)
        sync_select_from_idx()

    selected_uuid = st.selectbox(
        "选择 uuid",
        options=uuid_options,
        index=st.session_state.uuid_idx,
        key="uuid_select_value",
        on_change=on_select_change,
    )

    question_row = question_base[question_base["uuid"] == selected_uuid].iloc[0]
    records_by_model: list[tuple[str, pd.Series]] = []
    for label, _df, _desc in model_frames:
        model_record_df = filtered[(filtered["uuid"] == selected_uuid) & (filtered["model_name"] == label)]
        if model_record_df.empty:
            records_by_model.append((label, None))
        else:
            records_by_model.append((label, model_record_df.iloc[0]))

    # 调整宽度：题目栏更窄，模型栏更宽
    models_col, question_col = st.columns([3, 1])

    with question_col:
        st.markdown("#### 题目内容")
        with st.container(height=1200, border=True):
            render_question_block(question_row, lang)

    with models_col:
        st.markdown("#### 模型回答对比")
        # 最多四列并排展示（1~4 模型）
        cols = st.columns(len(records_by_model))
        for col, (label, model_row) in zip(cols, records_by_model):
            with col:
                st.markdown(f"**模型：{label}**")
                if model_row is None:
                    st.info("当前筛选条件下无记录。")
                    continue

                with st.container(height=1000, border=True):
                    meta_items = {k: model_row.get(k) for k in ["score", "finish_reason", "model", "language"]}
                    st.markdown("**元信息**")
                    st.json(meta_items)
                    st.divider()

                    if "llm_answer" in model_row:
                        st.markdown("**llm_answer**")
                        st.markdown(str(model_row.get("llm_answer")))
                        st.divider()
                    if "llm_reasoning" in model_row:
                        st.markdown("**llm_reasoning**")
                        st.markdown(str(model_row.get("llm_reasoning")))
                        st.divider()
                    if "llm_output" in model_row:
                        st.markdown("**llm_output**")
                        st.markdown(str(model_row.get("llm_output")))
                    st.divider()

                    eval_key = f"{selected_uuid}::{label}"
                    eval_placeholder = st.empty()
                    if st.session_state.get("eval_results", {}).get(eval_key):
                        eval_placeholder.markdown(st.session_state["eval_results"][eval_key])

                    if st.button("评估 reasoning+output", key=f"btn_eval_{selected_uuid}_{label}"):
                        try:
                            prompt_template = st.session_state.get("eval_prompt") or DEFAULT_EVAL_PROMPT
                            with st.spinner("调用 llm 评估中..."):
                                resp = stream_eval(prompt_template, question_row, model_row, lang, eval_placeholder)
                            st.session_state["eval_results"][eval_key] = resp
                        except Exception as exc:
                            eval_placeholder.error(f"评估失败: {exc}")


if __name__ == "__main__":
    main()

