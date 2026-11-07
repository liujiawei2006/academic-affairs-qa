(() => {
  "use strict";

  const form = document.getElementById("ask-form");
  const input = document.getElementById("question");
  const submit = document.getElementById("submit-question");
  const panel = document.getElementById("answer-panel");
  const status = document.getElementById("answer-status");
  const content = document.getElementById("answer-content");
  const sources = document.getElementById("sources");
  const sourceList = document.getElementById("source-list");
  const meta = document.getElementById("answer-meta");
  let controller = null;

  const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);

  function inline(text, validMarkers) {
    return escape(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d+)\]/g, (marker, n) => validMarkers.has(marker)
        ? `<button type="button" class="citation-link" data-source="source-${n}" aria-label="查看政策依据 ${n}">${marker}</button>` : marker);
  }

  function renderAnswer(text, citations) {
    const markers = new Set(citations.map((citation) => citation.marker));
    const lines = String(text || "").split(/\r?\n/);
    const blocks = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i].trim();
      if (!line) { i++; continue; }
      if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
        const rows = [];
        const cells = (row) => row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|");
        rows.push(`<thead><tr>${cells(line).map((c) => `<th>${inline(c.trim(), markers)}</th>`).join("")}</tr></thead><tbody>`);
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          rows.push(`<tr>${cells(lines[i]).map((c) => `<td>${inline(c.trim(), markers)}</td>`).join("")}</tr>`);
          i++;
        }
        blocks.push(`<div class="table-wrap"><table>${rows.join("")}</tbody></table></div>`);
      } else if (/^#{1,6}\s/.test(line)) {
        blocks.push(`<h3>${inline(line.replace(/^#{1,6}\s+/, ""), markers)}</h3>`);i++;
      } else if (/^(?:[-*]\s+|\d+[.、]\s*)/.test(line)) {
        const ordered = /^\d/.test(line);
        const items = [];
        const pattern = ordered ? /^\d+[.、]\s*/ : /^[-*]\s+/;
        while (i < lines.length && pattern.test(lines[i].trim())) {
          items.push(`<li>${inline(lines[i].trim().replace(pattern, ""), markers)}</li>`);i++;
        }
        const tag = ordered ? "ol" : "ul";
        blocks.push(`<${tag}>${items.join("")}</${tag}>`);
      } else {
        blocks.push(`<p>${inline(line, markers)}</p>`);i++;
      }
    }
    content.innerHTML = blocks.join("");
  }

  function renderSources(citations) {
    sourceList.replaceChildren();
    sources.hidden = !citations.length;
    for (const citation of citations) {
      const item = document.createElement("article");
      const number = /^\[(\d+)\]$/.exec(citation.marker)?.[1];
      item.className = "source-item";
      if (number) item.id = `source-${number}`;
      item.tabIndex = -1;
      const title = document.createElement("div");
      title.className = "source-title";
      title.textContent = `${citation.marker} ${citation.doc_title || citation.source_file}`;
      const location = document.createElement("div");
      location.className = "source-location";
      location.textContent = [citation.chapter, citation.article, citation.table_row].filter(Boolean).join(" · ");
      const file = document.createElement("div");
      file.className = "source-location";
      file.textContent = citation.source_file;
      item.append(title, location, file);
      sourceList.append(item);
    }
  }

  content.addEventListener("click", (event) => {
    const link = event.target.closest("[data-source]");
    if (!link) return;
    const source = document.getElementById(link.dataset.source);
    if (!source) return;
    sourceList.querySelectorAll(".highlighted").forEach((item) => item.classList.remove("highlighted"));
    source.classList.add("highlighted");
    source.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" });
    source.focus({ preventScroll: true });
  });

  document.getElementById("close-answer").addEventListener("click", () => {
    controller?.abort();
    panel.hidden = true;
    input.focus();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden) document.getElementById("close-answer").click();
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question || controller) return;
    panel.hidden = false;
    panel.scrollTop = 0;
    document.getElementById("asked-question").textContent = question;
    content.replaceChildren();sourceList.replaceChildren();sources.hidden = true;meta.textContent = "";
    status.className = "answer-status loading";
    status.textContent = "正在检索政策、生成回答并核对依据…";
    submit.disabled = true;
    form.setAttribute("aria-busy", "true");
    controller = new AbortController();
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true;controller?.abort(); }, 125000);
    try {
      if (location.protocol === "file:") throw new Error("请通过项目服务打开页面：运行 python run_api.py，然后访问 http://localhost:8000。");
      const response = await fetch("/qa", {
        method: "POST",headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, enable_verify: true }),signal: controller.signal,
      });
      if (!response.ok) {
        if (response.status === 503) throw new Error("问答服务尚未准备好，请稍后重试。");
        if (response.status === 504) throw new Error("回答超时，请稍后重试。");
        if (response.status === 404 || response.status === 405) throw new Error("当前仅为页面预览。要使用教务问答，请启动项目服务 python run_api.py 并访问 http://localhost:8000。");
        throw new Error(`暂时无法完成问答，请稍后重试（${response.status}）。`);
      }
      const result = await response.json();
      if (!result.answer) throw new Error("暂未生成有效回答，请稍后重试。");
      const citations = result.citations || [];
      renderAnswer(result.answer, citations);
      renderSources(citations);
      status.className = "answer-status";
      status.textContent = result.fallback?.should_fallback ? "未找到足够的政策依据，建议咨询教务处。" : "回答已生成 · 点击引用编号查看政策依据";
      const seconds = Object.values(result.timing || {}).reduce((sum, n) => sum + (Number(n) || 0), 0);
      meta.textContent = `${citations.length} 条政策引用${seconds ? ` · 用时 ${seconds.toFixed(1)}s` : ""}`;
    } catch (error) {
      status.className = "answer-status error";
      status.textContent = error.name === "AbortError"
        ? (timedOut ? "回答超时，请稍后重试。" : "本次请求已取消。")
        : error instanceof TypeError ? "无法连接问答服务，请确认项目服务已启动后重试。" : error.message;
    } finally {
      clearTimeout(timeout);controller = null;submit.disabled = false;
      form.removeAttribute("aria-busy");
    }
  });
})();
