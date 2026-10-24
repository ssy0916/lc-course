(function () {
  const messages = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const clearButton = document.getElementById("clear-history");
  const status = document.getElementById("status");
  const imageInput = document.getElementById("image-input");
  const imageButton = document.getElementById("image-button");
  const imagePreview = document.getElementById("image-preview");
  const imagePreviewThumb = document.getElementById("image-preview-thumb");
  const removeImageButton = document.getElementById("remove-image");
  const threadIdKey = "lc-course-thread-id";
  const threadId = sessionStorage.getItem(threadIdKey) || crypto.randomUUID();
  sessionStorage.setItem(threadIdKey, threadId);

  // 已上传到 OSS、等待随下一条消息发送的图片地址
  let pendingImageUrl = null;

  function contentText(content) {
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .map((part) => (typeof part === "string" ? part : part.text || ""))
        .filter(Boolean)
        .join("\n");
    }
    return "";
  }

  function addMessage(role, content) {
    const welcome = messages.querySelector(".welcome");
    if (welcome) welcome.remove();
    const article = document.createElement("article");
    article.className = `message ${role}`;
    article.setAttribute("aria-label", role === "user" ? "你" : "菜谱助手");
    article.textContent = contentText(content);
    messages.appendChild(article);
    messages.scrollTop = messages.scrollHeight;
    return article;
  }

  function handleFrame(frame, assistantMessage) {
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .join("\n");
    if (!data) return;

    const event = JSON.parse(data);
    if (event.type === "token") {
      assistantMessage.textContent += contentText(event.content);
      messages.scrollTop = messages.scrollHeight;
    } else if (event.type === "error") {
      throw new Error(event.message);
    }
  }

  async function loadHistory() {
    try {
      const response = await fetch(
        `/api/v1/chat/messages?thread_id=${encodeURIComponent(threadId)}`
      );
      if (!response.ok) throw new Error(`加载历史失败 (${response.status})`);
      const result = await response.json();
      if (result.messages.length) {
        messages.replaceChildren();
        result.messages.forEach((item) => addMessage(item.role, item.content));
      }
    } catch (error) {
      status.textContent = error.message;
    }
  }

  function fileExtension(name) {
    const dot = name.lastIndexOf(".");
    return dot === -1 ? "jpg" : name.slice(dot + 1).toLowerCase();
  }

  async function uploadToOss(file) {
    // 用随机 key 直传，避免同名文件互相覆盖
    const key = `uploads/${crypto.randomUUID()}.${fileExtension(file.name)}`;
    const presignResponse = await fetch(
      `/api/v1/oss/presign?filename=${encodeURIComponent(key)}`
    );
    if (!presignResponse.ok)
      throw new Error(`获取上传签名失败 (${presignResponse.status})`);
    const { uploadUrl, contentType, accessUrl } = await presignResponse.json();

    const uploadResponse = await fetch(uploadUrl, {
      method: "PUT",
      // Content-Type 必须与签名时一致，否则 OSS 会拒绝这次上传
      headers: { "Content-Type": contentType },
      body: file,
    });
    if (!uploadResponse.ok)
      throw new Error(`图片上传失败 (${uploadResponse.status})`);
    return accessUrl;
  }

  imageButton.addEventListener("click", () => imageInput.click());

  imageInput.addEventListener("change", async () => {
    const file = imageInput.files && imageInput.files[0];
    imageInput.value = "";
    if (!file) return;

    imageButton.disabled = true;
    status.textContent = "正在上传图片…";
    try {
      pendingImageUrl = await uploadToOss(file);
      imagePreviewThumb.src = pendingImageUrl;
      imagePreview.hidden = false;
      status.textContent = "图片已就绪，发送时会一起交给模型识别";
    } catch (error) {
      pendingImageUrl = null;
      imagePreviewThumb.removeAttribute("src");
      imagePreview.hidden = true;
      status.textContent = error.message;
    } finally {
      imageButton.disabled = false;
    }
  });

  removeImageButton.addEventListener("click", () => {
    pendingImageUrl = null;
    imagePreviewThumb.removeAttribute("src");
    imagePreview.hidden = true;
    status.textContent = "已移除待发送的图片";
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    const imageUrl = pendingImageUrl;
    addMessage("user", message);
    const assistantMessage = addMessage("assistant", "");
    input.value = "";
    pendingImageUrl = null;
    imagePreviewThumb.removeAttribute("src");
    imagePreview.hidden = true;
    input.disabled = true;
    sendButton.disabled = true;
    status.textContent = "正在思考…";

    try {
      const response = await fetch("/api/v1/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, image_url: imageUrl, thread_id: threadId }),
      });
      if (!response.ok) throw new Error(`请求失败 (${response.status})`);
      if (!response.body) throw new Error("浏览器不支持读取流式响应。");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          handleFrame(buffer.slice(0, boundary), assistantMessage);
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf("\n\n");
        }
        if (done) break;
      }
      if (buffer.trim()) handleFrame(buffer, assistantMessage);
      status.textContent = "回答完成";
    } catch (error) {
      status.textContent = error.message;
      if (!assistantMessage.textContent) assistantMessage.textContent = "回答生成失败。";
    } finally {
      input.disabled = false;
      sendButton.disabled = false;
      input.focus();
    }
  });

  clearButton.addEventListener("click", async () => {
    clearButton.disabled = true;
    try {
      const response = await fetch(
        `/api/v1/chat/messages?thread_id=${encodeURIComponent(threadId)}`,
        { method: "DELETE" }
      );
      if (!response.ok) throw new Error(`清空失败 (${response.status})`);
      messages.replaceChildren();
      const welcome = document.createElement("div");
      const greeting = document.createElement("strong");
      const prompt = document.createElement("span");
      welcome.className = "welcome";
      greeting.textContent = "你好，我是你的菜谱助手。";
      prompt.textContent = "试试问：“鸡蛋和番茄可以做什么？怎么做？”";
      welcome.append(greeting, prompt);
      messages.appendChild(welcome);
      status.textContent = "对话记录已清空";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      clearButton.disabled = false;
    }
  });

  loadHistory();
})();
